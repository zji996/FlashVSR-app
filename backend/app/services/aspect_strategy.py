from __future__ import annotations

from typing import Callable, Protocol, Tuple

from PIL import Image

from app.services.flashvsr_io import (
    compute_scaled_dims,
    upscale_and_crop,
    upscale_and_pad,
)
from app.services.video_export import VideoExporter


class AspectStrategy(Protocol):
    def build_transform(
        self,
        w0: int,
        h0: int,
        scale: float,
    ) -> Tuple[Callable[[Image.Image], Image.Image], int, int]:
        """返回 (per-frame transform, target_width, target_height)."""

    def finalize_output(
        self,
        path: str,
        source_width: int,
        source_height: int,
        scale: float,
    ) -> None:
        """可选的导出阶段处理（如裁剪），默认可为 no-op。"""


class CenterCrop128Strategy:
    """
    默认长宽比策略（preserve_aspect_ratio=False）。

    - 按 scale 放大；
    - 再向下对齐到 128 的倍数并居中裁剪。
    """

    def build_transform(
        self,
        w0: int,
        h0: int,
        scale: float,
    ) -> Tuple[Callable[[Image.Image], Image.Image], int, int]:
        sW, sH, tW, tH = compute_scaled_dims(w0, h0, scale)
        print(f"目标分辨率: {tW}x{tH} (缩放 {scale}x，裁剪自 {sW}x{sH})")

        def transform(img: Image.Image) -> Image.Image:
            return upscale_and_crop(img, scale, tW, tH)

        return transform, tW, tH

    def finalize_output(
        self,
        path: str,
        source_width: int,
        source_height: int,
        scale: float,
    ) -> None:
        # 直接居中裁剪的模式在预处理阶段即已完成，无需导出阶段二次裁剪。
        return None


class PadThenCropStrategy:
    """
    保持长宽比策略（preserve_aspect_ratio=True）。

    - 按 scale 放大内容到 (sW, sH)；
    - 再向上取整到 128 倍数 (tW, tH)，用黑边补足；
    - 导出阶段再按原始长宽比裁剪去黑边。
    """

    def build_transform(
        self,
        w0: int,
        h0: int,
        scale: float,
    ) -> Tuple[Callable[[Image.Image], Image.Image], int, int]:
        sW = int(round(w0 * scale))
        sH = int(round(h0 * scale))
        multiple = 128
        tW = ((sW + multiple - 1) // multiple) * multiple
        tH = ((sH + multiple - 1) // multiple) * multiple
        print(f"目标分辨率: {tW}x{tH} (缩放 {scale}x, padding 自 {sW}x{sH})")

        def transform(img: Image.Image) -> Image.Image:
            return upscale_and_pad(img, scale, sW, sH, tW, tH)

        return transform, tW, tH

    def finalize_output(
        self,
        path: str,
        source_width: int,
        source_height: int,
        scale: float,
    ) -> None:
        # 在 preserve_aspect_ratio=True 的模式下，导出阶段通过裁剪去除 padding 黑边，
        # 恢复按 scale 放大后的内容长宽比。
        exporter = VideoExporter()
        exporter.crop_to_aspect(
            path=path,
            source_width=source_width,
            source_height=source_height,
            scale=scale,
        )
