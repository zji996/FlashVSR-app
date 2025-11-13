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
    audio_path: Optional[Path] = None


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
        """始终执行预处理：容器规范化 + 缩放 + 音频分离."""

        target_width = params.preprocess_width
        if metadata.width and target_width and metadata.width <= target_width:
            logger.info(
                "源宽度 %s <= 目标宽度 %s，仍按统一流程进行容器规范化 + 尺寸对齐",
                metadata.width,
                target_width,
            )

        temp_path = self.tmp_dir / f"pre_{uuid4().hex}.mp4"
        audio_path = self.tmp_dir / f"pre_audio_{uuid4().hex}.m4a"

        logger.info(
            "FFmpeg 预处理开始: %s -> %s (width=%s)", input_path, temp_path, target_width
        )
        self._run_ffmpeg(input_path, temp_path, target_width)
        self._extract_audio(input_path, audio_path)
        logger.info("FFmpeg 预处理完成: %s -> %s (+ audio %s)", input_path, temp_path, audio_path)
        new_metadata = VideoMetadataService.extract_metadata(temp_path)
        return PreprocessResult(temp_path, True, new_metadata, audio_path if audio_path.exists() else None)

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
        """执行缩放/容器转换命令。

        - 支持 NVENC（h264_nvenc / hevc_nvenc）与 CPU（libx264 / libx265）。
        - 当选择 NVENC 但失败时，自动回退到 CPU 编码。
        - 统一像素格式为 yuv420p，强制偶数尺寸（-2）并写入 faststart。
        """

        def build_filter() -> list[str]:
            filters: list[str] = []
            if target_width:
                filters.append(f"scale={target_width}:-2")
            return ["-vf", ",".join(filters)] if filters else []

        def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.run(cmd, capture_output=True, text=True, check=False)

        hwaccel = (settings.PREPROCESS_FFMPEG_HWACCEL or "").strip().lower()
        vcodec = (settings.PREPROCESS_FFMPEG_VIDEO_CODEC or "libx264").strip()

        common_args = [
            settings.FFMPEG_BINARY,
            "-y",
        ]
        if hwaccel == "cuda":
            common_args += ["-hwaccel", "cuda"]

        input_args = ["-i", str(input_path)]
        filter_args = build_filter()
        out_common = [
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-map", "0:v:0",
            "-an",
            str(output_path),
        ]

        # Try preferred codec first (possibly NVENC)
        if vcodec in {"h264_nvenc", "hevc_nvenc"}:
            nvenc_args = [
                "-c:v", vcodec,
                "-preset", settings.PREPROCESS_NVENC_PRESET,
                "-rc", settings.PREPROCESS_NVENC_RC,
                "-cq", str(settings.PREPROCESS_NVENC_CQ),
            ]
            cmd = common_args + input_args + filter_args + nvenc_args + out_common
            result = run_cmd(cmd)
            if result.returncode == 0:
                return
            logger.warning("NVENC 预处理失败，回退到 CPU 编码: %s", result.stderr.strip())

        # CPU fallback or explicit CPU codec
        if vcodec in {"libx265", "hevc_nvenc"}:
            cpu_codec = "libx265"
        else:
            cpu_codec = "libx264"
        cpu_args = [
            "-c:v", cpu_codec,
            "-preset", settings.PREPROCESS_FFMPEG_PRESET,
            "-crf", str(settings.PREPROCESS_FFMPEG_CRF),
        ]
        cmd = common_args + input_args + filter_args + cpu_args + out_common
        result = run_cmd(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg 预处理失败（{result.returncode}）: {result.stderr.strip() or result.stdout.strip()}"
            )

    def _extract_audio(self, input_path: Path, audio_out: Path) -> None:
        """尝试无损提取音频；失败则转码为 AAC。无音频流时静默跳过。"""
        # First, probe if audio exists
        probe = subprocess.run(
            [
                settings.FFPROBE_BINARY,
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(input_path),
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0 or not probe.stdout.strip():
            logger.info("未检测到音频流，跳过音频提取")
            return

        # Try copy
        cmd_copy = [
            settings.FFMPEG_BINARY,
            "-y", "-i", str(input_path),
            "-map", "0:a:0", "-c:a", "copy", str(audio_out),
        ]
        res = subprocess.run(cmd_copy, capture_output=True, text=True)
        if res.returncode == 0:
            return

        # Fallback to AAC
        cmd_aac = [
            settings.FFMPEG_BINARY,
            "-y", "-i", str(input_path),
            "-map", "0:a:0", "-c:a", "aac", "-b:a", "192k", str(audio_out),
        ]
        res2 = subprocess.run(cmd_aac, capture_output=True, text=True)
        if res2.returncode != 0:
            logger.warning("音频提取失败（已忽略）：%s", res2.stderr.strip() or res2.stdout.strip())
