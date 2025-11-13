"""Chunked export utilities for FlashVSR."""

from __future__ import annotations

import time
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Callable, Optional, TYPE_CHECKING
from uuid import uuid4

import imageio
import numpy as np
import torch
from tqdm import tqdm

from app.config import settings

if TYPE_CHECKING:
    from app.services.flashvsr_service import FlashVSRService


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
        service: "FlashVSRService",
        output_path: str,
        fps: int,
        total_frames: int,
        start_time: Optional[float],
        progress_callback: Optional[Callable[[int, int, float], None]],
        audio_path: Optional[str] = None,
    ) -> None:
        self._service = service
        self.output_path = output_path
        self.total_frames = total_frames
        self.progress_callback = progress_callback
        self.start_time = start_time or time.time()
        self.processed = 0
        self.chunk_paths: list[Path] = []
        self.chunk_dir = settings.FLASHVSR_CHUNKED_SAVE_TMP_DIR / f"chunks_{uuid4().hex}"
        self.chunk_size = settings.FLASHVSR_CHUNKED_SAVE_CHUNK_SIZE
        self._buffer: list[np.ndarray] = []
        self.audio_path = audio_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = ChunkedVideoWriter(
            fps=fps,
            quality=6,
            chunk_dir=self.chunk_dir,
            base_name=Path(output_path).stem,
        )
        self._closed = False

    def handle_chunk(self, tensor_chunk: torch.Tensor) -> None:
        if tensor_chunk is None or tensor_chunk.shape[2] == 0:
            return
        chunk_cpu = tensor_chunk.detach().to("cpu")
        first_batch = chunk_cpu[0]
        frames = self._service._tensor2video(first_batch)
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
        self._service._merge_video_chunks(self.chunk_paths, self.output_path, audio_path=self.audio_path)
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
        self._service._cleanup_chunk_artifacts(self.chunk_paths)
        self._closed = True
        self._buffer.clear()

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
