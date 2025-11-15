from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    assert feat.dim() == 4, "feat 必须是 (N, C, H, W)"
    n, c = feat.shape[:2]
    var = feat.view(n, c, -1).var(dim=2, unbiased=False) + eps
    std = var.sqrt().view(n, c, 1, 1)
    mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return mean, std


def _adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    assert content_feat.shape[:2] == style_feat.shape[:2], "ADAIN: N、C 必须匹配"
    size = content_feat.size()
    style_mean, style_std = _calc_mean_std(style_feat)
    content_mean, content_std = _calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def _make_gaussian3x3_kernel(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    return torch.tensor(vals, dtype=dtype, device=device)


def _wavelet_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    assert x.dim() == 4, "x 必须是 (N, C, H, W)"
    _, c, _, _ = x.shape
    base = _make_gaussian3x3_kernel(x.dtype, x.device)
    weight = base.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    pad = radius
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    out = F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, dilation=radius, groups=c)
    return out


def _wavelet_decompose(x: torch.Tensor, levels: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 4, "x 必须是 (N, C, H, W)"
    high = torch.zeros_like(x)
    low = x
    for i in range(levels):
        radius = 2 ** i
        blurred = _wavelet_blur(low, radius)
        high = high + (low - blurred)
        low = blurred
    return high, low


def _wavelet_reconstruct(content: torch.Tensor, style: torch.Tensor, levels: int = 5) -> torch.Tensor:
    c_high, _ = _wavelet_decompose(content, levels=levels)
    _, s_low = _wavelet_decompose(style, levels=levels)
    return c_high + s_low


class TorchColorCorrectorWavelet(nn.Module):
    """视频友好的无状态颜色矫正模块（默认使用 wavelet 模式）。"""

    def __init__(self, levels: int = 5) -> None:
        super().__init__()
        self.levels = levels

    @staticmethod
    def _flatten_time(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        assert x.dim() == 5, "输入必须是 (B, C, f, H, W)"
        b, c, f, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        return y, b, f

    @staticmethod
    def _unflatten_time(y: torch.Tensor, b: int, f: int) -> torch.Tensor:
        bf, c, h, w = y.shape
        assert bf == b * f
        return y.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4)

    def forward(
        self,
        hq_image: torch.Tensor,  # (B, C, f, H, W)
        lq_image: torch.Tensor,  # (B, C, f, H, W)
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        method: Literal["wavelet", "adain"] = "wavelet",
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        assert hq_image.shape == lq_image.shape, "HQ 与 LQ 的形状必须一致"
        assert hq_image.dim() == 5 and hq_image.shape[1] == 3, "输入必须是 (B, 3, f, H, W)"

        b, c, f, h, w = hq_image.shape
        if chunk_size is None or chunk_size >= f:
            hq4, b, f = self._flatten_time(hq_image)
            lq4, _, _ = self._flatten_time(lq_image)
            if method == "wavelet":
                out4 = _wavelet_reconstruct(hq4, lq4, levels=self.levels)
            elif method == "adain":
                out4 = _adain(hq4, lq4)
            else:
                raise ValueError(f"未知 method: {method}")
            out4 = torch.clamp(out4, *clip_range)
            out = self._unflatten_time(out4, b, f)
            return out

        outs = []
        for start in range(0, f, chunk_size):
            end = min(start + chunk_size, f)
            hq_chunk = hq_image[:, :, start:end]
            lq_chunk = lq_image[:, :, start:end]
            hq4, b_, f_ = self._flatten_time(hq_chunk)
            lq4, _, _ = self._flatten_time(lq_chunk)
            if method == "wavelet":
                out4 = _wavelet_reconstruct(hq4, lq4, levels=self.levels)
            elif method == "adain":
                out4 = _adain(hq4, lq4)
            else:
                raise ValueError(f"未知 method: {method}")
            out4 = torch.clamp(out4, *clip_range)
            out_chunk = self._unflatten_time(out4, b_, f_)
            outs.append(out_chunk)
        out = torch.cat(outs, dim=2)
        return out


__all__ = ["TorchColorCorrectorWavelet"]

