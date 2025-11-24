from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

import torch

from app.config import settings
from app.services.chunk_export import ChunkedExportSession
from app.services.flashvsr_io import export_video_from_tensor, rescale_video_to_aspect
from app.services.progress import ProgressReporter


class VideoExporter:
    """High-level video export helper for FlashVSR outputs."""

    def __init__(self) -> None:
        # 当前实现不需要额外状态，但预留 settings 钩子以便未来扩展。
        self._settings = settings

    def _wrap_progress(
        self,
        reporter: Optional[ProgressReporter],
    ) -> Optional[Callable[[int, int, float], None]]:
        """
        Convert a ProgressReporter into the legacy (processed, total, avg) callback
        used by low-level IO helpers.
        """
        if reporter is None:
            return None

        last_processed = 0

        def _update(processed_frames: int, total_frames: int, avg_frame_time: float) -> None:
            nonlocal last_processed
            delta = max(processed_frames - last_processed, 0)
            last_processed = processed_frames
            reporter.report(delta)

        return _update

    def export_full_tensor(
        self,
        frames: torch.Tensor,  # (C, T, H, W)
        output_path: str,
        fps: int,
        progress: Optional[ProgressReporter],
        audio_path: Optional[str],
        total_frames: int,
        start_time: float,
    ) -> int:
        """
        Export a full in-memory tensor to the final MP4 file.

        Returns the number of frames written.
        """
        progress_callback = self._wrap_progress(progress)
        processed = export_video_from_tensor(
            output_video=frames,
            output_path=output_path,
            fps=fps,
            total_frames=total_frames,
            start_time=start_time,
            progress_callback=progress_callback,
            audio_path=audio_path,
        )
        if progress is not None:
            progress.done()
        return processed

    def create_chunk_session(
        self,
        output_path: str,
        fps: int,
        total_frames: int,
        progress: Optional[ProgressReporter],
        audio_path: Optional[str],
        start_time: float,
    ) -> ChunkedExportSession:
        """
        Create a chunked export session that writes intermediate MP4 chunks.
        """
        progress_callback = self._wrap_progress(progress)
        session = ChunkedExportSession(
            output_path=output_path,
            fps=fps,
            total_frames=total_frames,
            start_time=start_time,
            progress_callback=progress_callback,
            audio_path=audio_path,
        )
        return session

    def crop_to_aspect(
        self,
        path: str,
        source_width: int,
        source_height: int,
        scale: float,
    ) -> None:
        """
        Crop the final MP4 to match the scaled source aspect ratio.

        This is a thin wrapper around the existing rescale_video_to_aspect helper.
        """
        if not path or source_width <= 0 or source_height <= 0 or scale <= 0:
            return
        # 确保输出目录存在
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rescale_video_to_aspect(
            path=path,
            source_width=source_width,
            source_height=source_height,
            scale=scale,
        )

