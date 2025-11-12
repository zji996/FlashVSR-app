"""视频元数据解析服务."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import imageio

from app.config import settings


@dataclass
class VideoMetadata:
    """视频基础元数据."""

    width: Optional[int] = None
    height: Optional[int] = None
    total_frames: Optional[int] = None
    fps: Optional[int] = None
    duration: Optional[float] = None
    bit_rate: Optional[int] = None
    avg_frame_rate: Optional[float] = None

    def to_dict(self) -> dict[str, Optional[Union[int, float]]]:
        """转换为可序列化的字典."""
        return asdict(self)

    def pixels(self) -> Optional[int]:
        """返回分辨率像素数."""
        if self.width and self.height:
            return self.width * self.height
        return None

    def bits_per_pixel(self) -> Optional[float]:
        """估算比特/像素指标."""
        if not self.bit_rate:
            return None
        pixels = self.pixels()
        if not pixels:
            return None
        fps = self.fps or self.avg_frame_rate
        if not fps or fps <= 0:
            return self.bit_rate / pixels
        pixels_per_second = pixels * fps
        if pixels_per_second <= 0:
            return None
        return self.bit_rate / pixels_per_second


class VideoMetadataService:
    """视频元数据解析逻辑."""

    @staticmethod
    def extract_metadata(path: Union[str, Path]) -> VideoMetadata:
        """
        解析视频文件的基础元数据.

        Args:
            path: 视频文件路径

        Returns:
            VideoMetadata

        Raises:
            ValueError: 当文件不存在或无法解析时抛出
        """
        file_path = Path(path)
        if not file_path.exists():
            raise ValueError("视频文件不存在")

        metadata = VideoMetadata()
        reader = None
        reader_error: Optional[Exception] = None
        try:
            reader = imageio.get_reader(file_path.as_posix())
        except Exception as exc:
            reader_error = exc

        if reader is not None:
            try:
                # 提取元数据信息
                meta = {}
                try:
                    meta = reader.get_meta_data()
                except Exception:
                    meta = {}

                # 帧率
                fps = meta.get("fps")
                if isinstance(fps, (int, float)) and fps > 0:
                    metadata.fps = int(round(fps))

                # 帧数
                nframes = meta.get("nframes")
                if isinstance(nframes, int) and nframes > 0:
                    metadata.total_frames = nframes
                else:
                    try:
                        metadata.total_frames = reader.count_frames()
                    except Exception:
                        metadata.total_frames = None

                # 分辨率
                try:
                    frame0 = reader.get_data(0)
                    metadata.height, metadata.width = frame0.shape[:2]
                except Exception:
                    size = meta.get("size")
                    if isinstance(size, (list, tuple)) and len(size) == 2:
                        metadata.width = int(size[0])
                        metadata.height = int(size[1])

                # 时长
                if metadata.fps and metadata.total_frames:
                    metadata.duration = metadata.total_frames / metadata.fps
            finally:
                try:
                    reader.close()
                except Exception:
                    pass

        # 使用 ffprobe 补充或兜底
        try:
            VideoMetadataService._populate_ffprobe_info(file_path, metadata)
        except Exception:
            if reader is None:
                raise ValueError(f"无法读取视频文件: {reader_error}") from reader_error

        if metadata.width is None or metadata.height is None:
            raise ValueError("无法确定视频分辨率")

        return metadata

    @staticmethod
    def _populate_ffprobe_info(path: Path, metadata: VideoMetadata) -> None:
        """使用 ffprobe 获取码率等补充信息."""

        cmd = [
            settings.FFPROBE_BINARY,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=bit_rate,width,height,avg_frame_rate,nb_frames",
            "-show_entries",
            "format=bit_rate,duration",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout:
            return

        probe = json.loads(result.stdout)
        stream = None
        streams = probe.get("streams")
        if isinstance(streams, list) and streams:
            stream = streams[0]
        fmt = probe.get("format") or {}

        bit_rate = VideoMetadataService._parse_int(
            (stream or {}).get("bit_rate")
        ) or VideoMetadataService._parse_int(fmt.get("bit_rate"))
        if bit_rate:
            metadata.bit_rate = bit_rate

        width = VideoMetadataService._parse_int((stream or {}).get("width"))
        height = VideoMetadataService._parse_int((stream or {}).get("height"))
        if width and height:
            metadata.width = width
            metadata.height = height

        avg_frame_rate = VideoMetadataService._parse_ffprobe_fraction(
            (stream or {}).get("avg_frame_rate")
        )
        if avg_frame_rate:
            metadata.avg_frame_rate = avg_frame_rate
            if not metadata.fps:
                metadata.fps = int(round(avg_frame_rate))

        nb_frames = VideoMetadataService._parse_int((stream or {}).get("nb_frames"))
        if nb_frames and nb_frames > 0:
            metadata.total_frames = nb_frames

        duration = VideoMetadataService._parse_float(fmt.get("duration"))
        if duration and duration > 0:
            metadata.duration = duration
            if metadata.fps and not metadata.total_frames:
                metadata.total_frames = int(round(duration * metadata.fps))

    @staticmethod
    def _parse_int(value) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_ffprobe_fraction(value) -> Optional[float]:
        if not value:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and "/" in value:
            num, denom = value.split("/", 1)
            try:
                denom_val = float(denom)
                if denom_val == 0:
                    return None
                return float(num) / denom_val
            except ValueError:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_float(value) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
