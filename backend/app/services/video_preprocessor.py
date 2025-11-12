"""FFmpeg 预处理服务."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4

from app.config import settings
from app.schemas.task import TaskParameters
from app.services.video_metadata import VideoMetadata, VideoMetadataService

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """预处理结果."""

    input_path: Path
    applied: bool
    metadata: VideoMetadata


class VideoPreprocessor:
    """基于 FFmpeg 的预处理封装."""

    SAFE_EXTENSIONS: tuple[str, ...] = (
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".webm",
    )
    FORCE_TRANSCODE_EXTENSIONS: tuple[str, ...] = (
        ".ts",
        ".m2ts",
        ".mts",
        ".m4s",
        ".mpeg",
        ".mpg",
        ".vob",
    )

    def __init__(self) -> None:
        self.tmp_dir = settings.PREPROCESS_TMP_DIR
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def maybe_preprocess(
        self,
        input_path: Path,
        metadata: VideoMetadata,
        params: TaskParameters,
    ) -> PreprocessResult:
        """根据策略决定是否执行预处理."""

        strategy = params.preprocess_strategy
        target_width = params.preprocess_width

        needs_resize = strategy != "none" and bool(target_width)
        if needs_resize and metadata.width and target_width:
            if metadata.width <= target_width:
                logger.info(
                    "预处理跳过缩放：原视频宽度=%s 不大于目标宽度=%s",
                    metadata.width,
                    target_width,
                )
                needs_resize = False

        needs_container_fix = self._needs_container_normalization(Path(input_path))

        if not needs_resize and not needs_container_fix:
            logger.debug(
                "预处理跳过：策略=%s, 目标宽度=%s, 容器无需转换",
                strategy,
                target_width,
            )
            return PreprocessResult(input_path, False, metadata)

        temp_path = self.tmp_dir / f"pre_{uuid4().hex}.mp4"
        resize_width = target_width if needs_resize else None
        reason = []
        if needs_resize:
            reason.append("scale")
        if needs_container_fix:
            reason.append("container")
        logger.info(
            "FFmpeg 预处理开始（%s）: %s -> %s",
            "+".join(reason),
            input_path,
            temp_path,
        )
        self._run_ffmpeg(input_path, temp_path, resize_width)
        logger.info("FFmpeg 预处理完成: %s -> %s", input_path, temp_path)
        new_metadata = VideoMetadataService.extract_metadata(temp_path)
        return PreprocessResult(temp_path, True, new_metadata)

    def cleanup(self, path: Optional[Path]) -> None:
        """删除临时文件."""
        if not path:
            return
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    def _needs_container_normalization(self, input_path: Path) -> bool:
        """判断是否需要通过 FFmpeg 转换容器."""
        suffix = input_path.suffix.lower()
        if suffix in self.FORCE_TRANSCODE_EXTENSIONS:
            return True
        if suffix in self.SAFE_EXTENSIONS:
            return False
        # 其他未知扩展统一交给 FFmpeg 规范化，增加格式覆盖面
        return True

    def _run_ffmpeg(self, input_path: Path, output_path: Path, target_width: Optional[int]) -> None:
        """执行缩放/容器转换命令."""

        cmd = [
            settings.FFMPEG_BINARY,
            "-y",
            "-i",
            str(input_path),
        ]
        if target_width:
            vf = f"scale={target_width}:-2"
            cmd += ["-vf", vf]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            settings.PREPROCESS_FFMPEG_PRESET,
            "-crf",
            str(settings.PREPROCESS_FFMPEG_CRF),
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg 预处理失败（{result.returncode}）: {result.stderr.strip()}"
            )
