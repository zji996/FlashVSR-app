"""LQ 投影模块（Causal_LQ4x_Proj）。

从 upstream `examples/WanVSR/utils/utils.py` 精简移植，用于在后端本地
构建 LQ 投影层，而不依赖 third_party 运行时导入。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


CACHE_T = 2


def block_causal_mask(x: torch.Tensor, block_size: int) -> torch.Tensor:
    b, n, s, _, device = *x.size(), x.device
    assert s % block_size == 0
    num_blocks = s // block_size
    mask = torch.zeros(b, n, s, s, dtype=torch.bool, device=device)
    for i in range(num_blocks):
        mask[:, :, i * block_size : (i + 1) * block_size, : (i + 1) * block_size] = 1
    return mask


class RMS_norm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class CausalConv3d(nn.Conv3d):
    """Causal 3D 卷积，时间维前向因果。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding, mode="replicate")
        return super().forward(x)


class PixelShuffle3d(nn.Module):
    def __init__(self, ff: int, hh: int, ww: int) -> None:
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, H, W)
        return rearrange(
            x,
            "b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w",
            ff=self.ff,
            hh=self.hh,
            ww=self.ww,
        )


class Causal_LQ4x_Proj(nn.Module):
    """
    LQ 投影头，用于将低清视频特征编码到 WanVideo 语义空间。
    接收形状 (B, C, F, H, W)，输出一组线性层输出（list[Tensor]）。
    """

    def __init__(self, in_dim: int, out_dim: int, layer_num: int = 30) -> None:
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)

        self.conv1 = CausalConv3d(
            in_dim * self.ff * self.hh * self.ww,
            self.hidden_dim1,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3d(
            self.hidden_dim1,
            self.hidden_dim2,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()

        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)])

        self.clip_idx = 0
        self.cache: dict[str, Optional[torch.Tensor]] = {"conv1": None, "conv2": None}

    def clear_cache(self) -> None:
        self.cache = {"conv1": None, "conv2": None}
        self.clip_idx = 0

    def forward(self, video: torch.Tensor):
        self.clear_cache()
        # video: (B, C, F, H, W)
        t = video.shape[2]
        iter_ = 1 + (t - 1) // 4
        first_frame = video[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)

        out_x = []
        for i in range(iter_):
            x = self.pixel_shuffle(video[:, :, i * 4 : (i + 1) * 4, :, :])
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache["conv1"])
            self.cache["conv1"] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            if i == 0:
                self.cache["conv2"] = cache2_x
                continue
            x = self.conv2(x, self.cache["conv2"])
            self.cache["conv2"] = cache2_x
            x = self.norm2(x)
            x = self.act2(x)
            out_x.append(x)
        out_x_cat = torch.cat(out_x, dim=2)
        out_x_cat = rearrange(out_x_cat, "b c f h w -> b (f h w) c")
        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](out_x_cat))
        return outputs

    def stream_forward(self, video_clip: torch.Tensor) -> Optional[list[torch.Tensor]]:
        """
        流式前向接口，按 clip 逐段处理 LQ 视频。

        与 upstream Causal_LQ4x_Proj.stream_forward 保持一致，实现缓存复用：
        - 第一次调用只更新缓存，不返回任何投影（返回 None）
        - 后续调用在更新缓存的同时返回当前 clip 的线性层输出列表
        """
        if self.clip_idx == 0:
            # 首个 clip：在时间维前面补 3 帧首帧，建立缓存但不输出
            first_frame = video_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache["conv1"])
            self.cache["conv1"] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache["conv2"] = cache2_x
            self.clip_idx += 1
            return None
        else:
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache["conv1"])
            self.cache["conv1"] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv2(x, self.cache["conv2"])
            self.cache["conv2"] = cache2_x
            x = self.norm2(x)
            x = self.act2(x)
            out_x = rearrange(x, "b c f h w -> b (f h w) c")
            outputs: list[torch.Tensor] = []
            for i in range(self.layer_num):
                outputs.append(self.linear_layers[i](out_x))
            self.clip_idx += 1
            return outputs
