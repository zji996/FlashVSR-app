"""Streaming helpers for FlashVSR LQ buffers."""

from __future__ import annotations

import logging
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import islice
from threading import Condition, Thread
from typing import Callable, Deque, Optional
import numpy as np

import torch

logger = logging.getLogger(__name__)


def _bytes_to_gib(value: int) -> float:
    if value <= 0:
        return 0.0
    return value / float(1024 ** 3)


class FrameLoaderCoordinator:
    """Coordinate multi-threaded frame decoding into the streaming buffer."""

    def __init__(
        self,
        owner: "StreamingVideoTensor",
        indices: list[int],
        read_frame_fn: Callable[[int], np.ndarray],
        process_frame_fn: Callable[[np.ndarray], torch.Tensor],
        decode_workers: int,
        finalize_reader: Callable[[], None],
    ) -> None:
        self._owner = owner
        self._indices = indices
        self._read_frame_fn = read_frame_fn
        self._process_frame_fn = process_frame_fn
        self._decode_workers = max(1, decode_workers)
        self._finalize_reader = finalize_reader
        self._thread = Thread(target=self._run, name="flashvsr-lq-stream", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _run(self) -> None:
        pending_futures: Deque[Future[torch.Tensor]] = deque()
        total = len(self._indices)
        submitted = 0
        consumed = 0
        executor = ThreadPoolExecutor(
            max_workers=self._decode_workers, thread_name_prefix="flashvsr-lq-pre"
        )
        try:
            while consumed < total and not self._owner.should_stop:
                # Submit as many decode jobs as capacity allows.
                made_submission = False
                while submitted < total and not self._owner.should_stop:
                    if not self._owner.try_reserve_slot():
                        break
                    frame_idx = self._indices[submitted]
                    # Read from imageio reader sequentially in this thread.
                    frame_array = self._read_frame_fn(frame_idx)
                    # CPU preprocessing in worker threads.
                    pending_futures.append(executor.submit(self._process_frame_fn, frame_array))
                    submitted += 1
                    made_submission = True

                if not pending_futures:
                    if self._owner.should_stop or submitted >= total:
                        break
                    if not made_submission:
                        if not self._owner.wait_for_capacity():
                            break
                    continue

                future = pending_futures[0]
                try:
                    frame_tensor = future.result()
                except BaseException as exc:
                    self._owner.release_reserved_slot()
                    raise exc
                else:
                    self._owner.on_loader_frame_ready(frame_tensor)
                    consumed += 1
                finally:
                    pending_futures.popleft()
        except BaseException as exc:  # pragma: no cover - best effort logging
            if not self._owner.should_stop:
                logger.exception("LQ 流式缓冲加载线程异常", exc_info=exc)
            self._owner.on_loader_failed(exc)
        finally:
            while pending_futures:
                future = pending_futures.popleft()
                future.cancel()
                self._owner.release_reserved_slot()
            executor.shutdown(wait=False, cancel_futures=True)
            try:
                self._finalize_reader()
            except Exception:  # pragma: no cover - best effort logging
                logger.exception("关闭视频读取器失败")
            self._owner.on_loader_finished()


class StreamingVideoTensor:
    """In-memory bounded buffer that decodes frames asynchronously."""

    def __init__(
        self,
        reader,
        indices: list[int],
        read_frame_fn: Callable[[int], np.ndarray],
        process_frame_fn: Callable[[np.ndarray], torch.Tensor],
        height: int,
        width: int,
        dtype: torch.dtype,
        max_buffer_frames: int,
        prefetch_frames: int,
        per_frame_bytes: int,
        target_device: str,
        decode_workers: int,
        lock_memory: bool,
    ):
        self.height = height
        self.width = width
        self.dtype = dtype
        self.total_frames = len(indices)
        self._reader = reader
        self._read_frame_fn = read_frame_fn
        self._process_frame_fn = process_frame_fn
        self._max_buffer_frames = max_buffer_frames
        self._prefetch_frames = min(prefetch_frames, self.total_frames)
        self._per_frame_bytes = per_frame_bytes
        self._lock_memory = lock_memory and max_buffer_frames > 0
        self._capacity_bytes = max_buffer_frames * per_frame_bytes
        self._buffer: Deque[torch.Tensor] = deque()
        self._free_slots: Deque[torch.Tensor] = deque()
        self._target_device = target_device
        self._decode_workers = max(1, decode_workers)
        self._lock = Condition()
        self._start_index = 0
        self._produced = 0
        self._scheduled_slots = 0
        self._stop = False
        self._finished = False
        self._reader_closed = False
        self._error: Optional[BaseException] = None
        self._slot_storage: Optional[torch.Tensor] = None
        self._init_slot_pool()
        self._log_profile()
        self._loader: Optional[FrameLoaderCoordinator] = FrameLoaderCoordinator(
            owner=self,
            indices=indices,
            read_frame_fn=self._read_frame_fn,
            process_frame_fn=self._process_frame_fn,
            decode_workers=self._decode_workers,
            finalize_reader=self._close_reader,
        )
        self._loader.start()

        if self._prefetch_frames:
            self._wait_for_frames(self._prefetch_frames)

    @property
    def should_stop(self) -> bool:
        return self._stop

    def _init_slot_pool(self) -> None:
        if not self._lock_memory:
            return

        pin_memory = self._target_device == "cpu" and torch.cuda.is_available()
        self._slot_storage = torch.empty(
            (
                self._max_buffer_frames,
                3,
                self.height,
                self.width,
            ),
            dtype=self.dtype,
            device=self._target_device,
            pin_memory=pin_memory if self._target_device == "cpu" else False,
        )
        for idx in range(self._max_buffer_frames):
            self._free_slots.append(self._slot_storage[idx])

    def _log_profile(self) -> None:
        logger.info(
            "LQ 流式缓冲：容量 %d 帧 (≈%.2f GiB), 预读 %d 帧, 解码线程 %d, 预锁内存=%s",
            self._max_buffer_frames,
            _bytes_to_gib(self._capacity_bytes),
            self._prefetch_frames,
            self._decode_workers,
            "是" if self._lock_memory else "否",
        )
        if self._max_buffer_frames < self.total_frames:
            logger.warning(
                "处理帧数: %d，一直达到能处理的最多帧数",
                self._max_buffer_frames,
            )
        else:
            logger.info("处理帧数: %d", self.total_frames)

    def try_reserve_slot(self) -> bool:
        with self._lock:
            if self._stop:
                return False
            if self._scheduled_slots >= self._max_buffer_frames:
                return False
            self._scheduled_slots += 1
            return True

    def release_reserved_slot(self, count: int = 1) -> None:
        with self._lock:
            self._release_reserved_slot_locked(count)

    def _release_reserved_slot_locked(self, count: int) -> None:
        if count <= 0:
            return
        self._scheduled_slots = max(0, self._scheduled_slots - count)
        self._lock.notify_all()

    def wait_for_capacity(self) -> bool:
        with self._lock:
            while not self._stop and self._scheduled_slots >= self._max_buffer_frames:
                self._lock.wait()
            return not self._stop

    def on_loader_frame_ready(self, frame_tensor: torch.Tensor) -> None:
        with self._lock:
            if self._stop:
                self._release_reserved_slot_locked(1)
                return
            if self._lock_memory:
                if not self._free_slots:
                    raise RuntimeError("预锁定槽已耗尽，无法存储更多 LQ 帧")
                slot = self._free_slots.popleft()
                slot.copy_(frame_tensor)
                stored = slot
            else:
                stored = frame_tensor
            self._buffer.append(stored)
            self._produced += 1
            self._lock.notify_all()

    def on_loader_failed(self, exc: BaseException) -> None:
        with self._lock:
            self._error = exc
            self._stop = True
            self._lock.notify_all()

    def on_loader_finished(self) -> None:
        with self._lock:
            self._finished = True
            self._lock.notify_all()

    def _close_reader(self) -> None:
        if self._reader_closed:
            return
        self._reader_closed = True
        try:
            self._reader.close()
        except Exception:
            pass

    def _wait_for_frames(self, count: int) -> None:
        with self._lock:
            while self._produced < count and not self._error and not self._finished:
                self._lock.wait()
            self._raise_if_failed_locked()

    def _raise_if_failed_locked(self) -> None:
        if self._error:
            raise RuntimeError("LQ streaming loader failed") from self._error

    def get_clip(
        self,
        start: int,
        end: int,
        *,
        device: str,
        dtype: torch.dtype,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        start = max(start, 0)
        end = min(end, self.total_frames)
        if end <= start:
            return torch.empty((1, 3, 0, self.height, self.width), device=device, dtype=dtype)

        with self._lock:
            while self._produced < end and not self._error:
                if self._finished and self._produced >= end:
                    break
                self._lock.wait()
            self._raise_if_failed_locked()
            if end > self._produced:
                raise RuntimeError(
                    f"Requested frames [{start}, {end}) but only {self._produced} decoded so far"
                )
            if start < self._start_index:
                raise RuntimeError(
                    f"LQ frames [{start}, {end}) 已被释放，当前缓冲起点为 {self._start_index}"
                )
            relative_start = start - self._start_index
            relative_end = end - self._start_index
            frames = list(islice(self._buffer, relative_start, relative_end))

        if not frames:
            return torch.empty((1, 3, 0, self.height, self.width), device=device, dtype=dtype)

        clip = torch.stack(frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        return clip.to(device=device, dtype=dtype, non_blocking=non_blocking)

    def release_until(self, frame_idx: int) -> None:
        with self._lock:
            target = min(frame_idx, self._produced)
            if target <= self._start_index:
                return
            to_drop = min(target - self._start_index, len(self._buffer))
            for _ in range(to_drop):
                slot = self._buffer.popleft()
                if self._lock_memory:
                    self._free_slots.append(slot)
                self._start_index += 1
            self._release_reserved_slot_locked(to_drop)
            self._lock.notify_all()

    def cleanup(self) -> None:
        with self._lock:
            self._stop = True
            self._lock.notify_all()
        if self._loader:
            self._loader.join(timeout=5)
        with self._lock:
            self._buffer.clear()
            self._free_slots.clear()
            self._scheduled_slots = 0
            self._finished = True
            self._lock.notify_all()
        self._close_reader()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
