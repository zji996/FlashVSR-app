"""视频元数据解析服务."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import imageio


@dataclass
class VideoMetadata:
    """视频基础元数据."""

    width: Optional[int] = None
    height: Optional[int] = None
    total_frames: Optional[int] = None
    fps: Optional[int] = None
    duration: Optional[float] = None

    def to_dict(self) -> dict[str, Optional[Union[int, float]]]:
        """转换为可序列化的字典."""
        return asdict(self)


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

        try:
            reader = imageio.get_reader(file_path.as_posix())
        except Exception as exc:
            raise ValueError(f"无法读取视频文件: {exc}") from exc

        try:
            metadata = VideoMetadata()

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

            return metadata
        finally:
            try:
                reader.close()
            except Exception:
                pass
