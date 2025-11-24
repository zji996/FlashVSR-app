"""Chunked export utilities for FlashVSR."""

from __future__ import annotations

import time
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Callable, Optional
from uuid import uuid4

import imageio
import numpy as np
import torch
from tqdm import tqdm

from app.config import settings
from app.services.flashvsr_io import (
    tensor_to_video,
    merge_video_chunks,
    cleanup_chunk_artifacts,
)


def build_chunk_base_name(output_path: str, max_length: int = 64) -> str:
    """
    根据最终输出路径生成分片文件的基础文件名，限制长度以避免
    “File name too long” 等文件系统限制问题。
    """
    stem = Path(output_path).stem or "output"
    safe_chars = []
    for ch in stem:
        if ch.isalnum() or ch in {"-", "_"}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe_stem = "".join(safe_chars).strip("_") or "output"
    if len(safe_stem) > max_length:
        import hashlib

        digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
        safe_stem = f"{safe_stem[: max_length - 9]}_{digest}"
    return safe_stem


class ChunkedVideoWriter:
    """Background process that writes frame batches into chunk MP4 files."""

    def __init__(self, fps: int, quality: int, chunk_dir: Path, base_name: str) -> None:
        self.fps = fps
        self.quality = quality
        self.chunk_dir = chunk_dir
        self.base_name = base_name
        self._queue: Queue[dict[str, Any]] = Queue(maxsize=2)
        self._thread: Optional[Thread] = None
        self._started = False
        self._closed = False
        self._error: Optional[str] = None

    def _run(self) -> None:
        try:
            while True:
                message = self._queue.get()
                if message["type"] == "stop":
                    break
                path = Path(message["path"])
                frames = message["frames"]
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(str(path), fps=self.fps, quality=self.quality)
                try:
                    for frame in frames:
                        writer.append_data(frame)
                finally:
                    writer.close()
        except Exception as exc:  # pragma: no cover - best effort logging
            self._error = str(exc)

    def _start_worker(self) -> None:
        if self._started:
            return
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self._thread = Thread(target=self._run, name="chunk-writer", daemon=True)
        self._thread.start()
        self._started = True

    def submit(self, index: int, frames: list[np.ndarray]) -> Path:
        """Submit a chunk and return the chunk path."""
        if self._error:
            raise RuntimeError(f"分片写入线程失败: {self._error}")
        self._start_worker()
        chunk_path = self.chunk_dir / f"{self.base_name}_chunk_{index:05d}.mp4"
        self._queue.put(
            {
                "type": "chunk",
                "path": str(chunk_path),
                "frames": frames,
            }
        )
        if self._error:
            raise RuntimeError(f"分片写入线程失败: {self._error}")
        return chunk_path

    def finish(self) -> None:
        """Wait for the worker process to flush all pending chunks."""
        if not self._started or self._closed:
            self._closed = True
            return
        self._queue.put({"type": "stop"})
        if self._thread is not None:
            self._thread.join()
        self._closed = True
        if self._error:
            raise RuntimeError(f"分片写入线程失败: {self._error}")

    def abort(self) -> None:
        """Terminate the worker process if it's still alive."""
        if self._closed:
            return
        if self._started and self._thread is not None and self._thread.is_alive():
            self._queue.put({"type": "stop"})
            self._thread.join(timeout=1)
        self._closed = True


class ChunkedExportSession:
    """Manage chunked frame export and final merge."""

    def __init__(
        self,
        output_path: str,
        fps: int,
        total_frames: int,
        start_time: Optional[float],
        progress_callback: Optional[Callable[[int, int, float], None]],
        audio_path: Optional[str] = None,
    ) -> None:
        self.output_path = output_path
        self.total_frames = total_frames
        self.progress_callback = progress_callback
        self.start_time = start_time or time.time()
        self.processed = 0
        self.chunk_paths: list[Path] = []
        self.chunk_dir = settings.FLASHVSR_CHUNKED_SAVE_TMP_DIR / f"chunks_{uuid4().hex}"

        # 当环境变量设置为 0 时，按视频总帧数自动估算一个保守的分片大小，
        # 目标是生成数量适中的分片（约 8～16 个），避免过多小文件或单个巨型分片。
        base_chunk = settings.FLASHVSR_CHUNKED_SAVE_CHUNK_SIZE
        if base_chunk <= 0 and total_frames and total_frames > 0:
            # 至少 60 帧一个分片，最多约 total_frames / 4，防止极端值。
            target_chunks = 12
            auto_size = max(total_frames // target_chunks, 60)
            upper_bound = max(total_frames // 4, 60)
            self.chunk_size = min(auto_size, upper_bound)
        else:
            self.chunk_size = base_chunk

        chunk_base_name = build_chunk_base_name(output_path)
        self._buffer: list[np.ndarray] = []
        self.audio_path = audio_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = ChunkedVideoWriter(
            fps=fps,
            quality=settings.FLASHVSR_EXPORT_VIDEO_QUALITY,
            chunk_dir=self.chunk_dir,
            base_name=chunk_base_name,
        )
        self._closed = False
        self._partial_path = Path(output_path).with_name(
            f"{Path(output_path).stem}_partial{Path(output_path).suffix}"
        )

    def handle_chunk(self, tensor_chunk: torch.Tensor) -> None:
        if tensor_chunk is None or tensor_chunk.shape[2] == 0:
            return
        chunk_cpu = tensor_chunk.detach().to("cpu")
        first_batch = chunk_cpu[0]
        frames = tensor_to_video(first_batch)
        frame_arrays = [np.array(frame) for frame in frames]
        self._buffer.extend(frame_arrays)
        self._drain_buffer()
        self.processed += len(frame_arrays)
        if self.progress_callback:
            elapsed = max(time.time() - self.start_time, 0.0)
            avg_time = elapsed / self.processed if self.processed else 0.0
            self.progress_callback(self.processed, self.total_frames, avg_time)

    def close(self) -> None:
        if self._closed:
            return
        self._flush_buffer()
        self.writer.finish()
        merge_video_chunks(self.chunk_paths, self.output_path, audio_path=self.audio_path)
        if self.progress_callback:
            elapsed = max(time.time() - self.start_time, 0.0)
            avg_time = elapsed / self.total_frames if self.total_frames else 0.0
            self.progress_callback(self.total_frames, self.total_frames, avg_time)
        self.chunk_paths.clear()
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        self.writer.abort()
        cleanup_chunk_artifacts(self.chunk_paths)
        self._closed = True
        self._buffer.clear()
        self.chunk_paths.clear()

    def finalize_partial(self) -> Optional[Path]:
        """尝试输出已完成的部分结果."""
        if self._closed:
            return None
        has_data = self.processed > 0 or bool(self._buffer) or bool(self.chunk_paths)
        if not has_data:
            return None
        self._flush_buffer()
        try:
            self.writer.finish()
        except Exception:
            cleanup_chunk_artifacts(self.chunk_paths)
            self._buffer.clear()
            self.chunk_paths.clear()
            self._closed = True
            raise
        partial_path = self._partial_path
        try:
            merge_video_chunks(self.chunk_paths, str(partial_path), audio_path=self.audio_path)
        except Exception:
            cleanup_chunk_artifacts(self.chunk_paths)
            self._buffer.clear()
            self.chunk_paths.clear()
            self._closed = True
            raise
        self._buffer.clear()
        self.chunk_paths.clear()
        self._closed = True
        return partial_path

    def _drain_buffer(self) -> None:
        if self.chunk_size <= 0:
            self._flush_buffer()
            return
        while len(self._buffer) >= self.chunk_size:
            self._flush_buffer(self.chunk_size)

    def _flush_buffer(self, count: Optional[int] = None) -> None:
        if not self._buffer:
            return
        if count is None or count > len(self._buffer):
            count = len(self._buffer)
        if count <= 0:
            return
        frames_to_write = self._buffer[:count]
        del self._buffer[:count]
        self.chunk_paths.append(
            self.writer.submit(len(self.chunk_paths), frames_to_write)
        )
