import types
import os
import time
from typing import Optional, Tuple, Literal, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from PIL import Image
from tqdm import tqdm
# import pyfiglet

from ..models import ModelManager
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d, build_3d_freqs
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline


# -----------------------------
# 基础工具：ADAIN 所需的统计量（保留以备需要；管线默认用 wavelet）
# -----------------------------
def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    assert feat.dim() == 4, 'feat 必须是 (N, C, H, W)'
    N, C = feat.shape[:2]
    var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


def _adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    assert content_feat.shape[:2] == style_feat.shape[:2], "ADAIN: N、C 必须匹配"
    size = content_feat.size()
    style_mean, style_std = _calc_mean_std(style_feat)
    content_mean, content_std = _calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


# -----------------------------
# 小波式模糊与分解/重构（ColorCorrector 用）
# -----------------------------
def _make_gaussian3x3_kernel(dtype, device) -> torch.Tensor:
    vals = [
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125 ],
        [0.0625, 0.125, 0.0625],
    ]
    return torch.tensor(vals, dtype=dtype, device=device)


def _wavelet_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    assert x.dim() == 4, 'x 必须是 (N, C, H, W)'
    N, C, H, W = x.shape
    base = _make_gaussian3x3_kernel(x.dtype, x.device)
    weight = base.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    pad = radius
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    out = F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, dilation=radius, groups=C)
    return out


def _wavelet_decompose(x: torch.Tensor, levels: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 4, 'x 必须是 (N, C, H, W)'
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


# -----------------------------
# 无状态颜色矫正模块（视频友好，默认 wavelet）
# -----------------------------
class TorchColorCorrectorWavelet(nn.Module):
    def __init__(self, levels: int = 5):
        super().__init__()
        self.levels = levels

    @staticmethod
    def _flatten_time(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        assert x.dim() == 5, '输入必须是 (B, C, f, H, W)'
        B, C, f, H, W = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(B * f, C, H, W)
        return y, B, f

    @staticmethod
    def _unflatten_time(y: torch.Tensor, B: int, f: int) -> torch.Tensor:
        BF, C, H, W = y.shape
        assert BF == B * f
        return y.reshape(B, f, C, H, W).permute(0, 2, 1, 3, 4)

    def forward(
        self,
        hq_image: torch.Tensor,  # (B, C, f, H, W)
        lq_image: torch.Tensor,  # (B, C, f, H, W)
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        method: Literal['wavelet', 'adain'] = 'wavelet',
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        assert hq_image.shape == lq_image.shape, "HQ 与 LQ 的形状必须一致"
        assert hq_image.dim() == 5 and hq_image.shape[1] == 3, "输入必须是 (B, 3, f, H, W)"

        B, C, f, H, W = hq_image.shape
        if chunk_size is None or chunk_size >= f:
            hq4, B, f = self._flatten_time(hq_image)
            lq4, _, _ = self._flatten_time(lq_image)
            if method == 'wavelet':
                out4 = _wavelet_reconstruct(hq4, lq4, levels=self.levels)
            elif method == 'adain':
                out4 = _adain(hq4, lq4)
            else:
                raise ValueError(f"未知 method: {method}")
            out4 = torch.clamp(out4, *clip_range)
            out = self._unflatten_time(out4, B, f)
            return out

        outs = []
        for start in range(0, f, chunk_size):
            end = min(start + chunk_size, f)
            hq_chunk = hq_image[:, :, start:end]
            lq_chunk = lq_image[:, :, start:end]
            hq4, B_, f_ = self._flatten_time(hq_chunk)
            lq4, _, _ = self._flatten_time(lq_chunk)
            if method == 'wavelet':
                out4 = _wavelet_reconstruct(hq4, lq4, levels=self.levels)
            elif method == 'adain':
                out4 = _adain(hq4, lq4)
            else:
                raise ValueError(f"未知 method: {method}")
            out4 = torch.clamp(out4, *clip_range)
            out_chunk = self._unflatten_time(out4, B_, f_)
            outs.append(out_chunk)
        out = torch.cat(outs, dim=2)
        return out


# -----------------------------
# 简化版 Pipeline（仅 dit + vae）
# -----------------------------
class FlashVSRTinyLongPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False
        self.prompt_emb_posi = None
        self.ColorCorrector = TorchColorCorrectorWavelet(levels=5)

        print(r"""
███████╗██╗      █████╗ ███████╗██╗  ██╗██╗   ██╗███████╗█████╗
██╔════╝██║     ██╔══██╗██╔════╝██║  ██║██║   ██║██╔════╝██╔══██╗
█████╗  ██║     ███████║███████╗███████║╚██╗ ██╔╝███████╗███████║
██╔══╝  ██║     ██╔══██║╚════██║██╔══██║ ╚████╔╝ ╚════██║██╔═██║
██║     ███████╗██║  ██║███████║██║  ██║  ╚██╔╝  ███████║██║  ██║
╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
                         ⚡FlashVSR
""")

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        # 仅管理 dit / vae
        dtype = next(iter(self.dit.parameters())).dtype
        from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager: ModelManager):
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = FlashVSRTinyLongPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        # 可选：统一序列并行入口（此处默认关闭）
        pipe.use_unified_sequence_parallel = False
        return pipe

    def denoising_model(self):
        return self.dit

    # ---- Pipeline parallel placement ----
    def enable_pipeline_parallel(self, devices: list[str], split_index: Optional[int] = None):
        super().enable_pipeline_parallel(devices, split_index)
        if not devices or len(devices) < 2:
            return self
        dev0 = devices[0]
        dev1 = devices[-1]
        # Default split at middle if not provided
        if split_index is None:
            split_index = len(self.dit.blocks) // 2 - 1
            split_index = max(0, min(split_index, len(self.dit.blocks) - 2))
        self.pp_split_idx = split_index
        # Move patch embedding to first stage, head to last stage
        try:
            self.dit.patch_embedding.to(dev0)
        except Exception:
            pass
        try:
            self.dit.head.to(dev1)
        except Exception:
            pass
        # Place blocks and migrate cross-attn KV caches if present
        for i, blk in enumerate(self.dit.blocks):
            target = dev0 if i <= split_index else dev1
            try:
                blk.to(target)
            except Exception:
                pass
            # Move persistent cross-attention cache to the same device
            try:
                ca = blk.cross_attn
                if hasattr(ca, "cache_k") and ca.cache_k is not None:
                    ca.cache_k = ca.cache_k.to(target)
                if hasattr(ca, "cache_v") and ca.cache_v is not None:
                    ca.cache_v = ca.cache_v.to(target)
            except Exception:
                pass
        # Keep pipeline default device as first stage for VAE and utilities
        self.device = dev0
        return self

    # -------------------------
    # 新增：显式 KV 预初始化函数
    # -------------------------
    def init_cross_kv(
        self,
        context_tensor: Optional[torch.Tensor] = None,
    ):
        self.load_models_to_device(["dit"])
        """
        使用固定 prompt 生成文本 context，并在 WanModel 中初始化所有 CrossAttention 的 KV 缓存。
        必须在 __call__ 前显式调用一次。
        """
        prompt_path = "../../examples/WanVSR/prompt_tensor/posi_prompt.pth"

        if self.dit is None:
            raise RuntimeError("请先通过 fetch_models / from_model_manager 初始化 self.dit")

        if context_tensor is None:
            if prompt_path is None:
                raise ValueError("init_cross_kv: 需要提供 prompt_path 或 context_tensor 其一")
            ctx = torch.load(prompt_path, map_location=self.device)
        else:
            ctx = context_tensor

        ctx = ctx.to(dtype=self.torch_dtype, device=self.device)

        if self.prompt_emb_posi is None:
            self.prompt_emb_posi = {}
        self.prompt_emb_posi['context'] = ctx

        if hasattr(self.dit, "reinit_cross_kv"):
            self.dit.reinit_cross_kv(ctx)
        else:
            raise AttributeError("WanModel 缺少 reinit_cross_kv(ctx) 方法，请在模型实现中加入该能力。")
        self.timestep = torch.tensor([1000.], device=self.device, dtype=self.torch_dtype)
        self.t = self.dit.time_embedding(sinusoidal_embedding_1d(self.dit.freq_dim, self.timestep))
        self.t_mod = self.dit.time_projection(self.t).unflatten(1, (6, self.dit.dim))
        # Scheduler
        self.scheduler.set_timesteps(1, denoising_strength=1.0, shift=5.0)
        self.load_models_to_device([])

    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}

    def prepare_extra_input(self, latents=None):
        return {}

    @staticmethod
    def _suggest_color_chunk_size(height: int, width: int) -> Optional[int]:
        pixels = height * width
        if pixels >= 2560 * 1440:
            return 4
        if pixels >= 1920 * 1080:
            return 6
        return None

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        negative_prompt="",
        denoising_strength=1.0,
        seed=None,
        rand_device="gpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(60, 104),
        tile_stride=(30, 52),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="Wan2.1-T2V-14B",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        LQ_video=None,
        is_full_block=False,
        if_buffer=False,
        topk_ratio=2.0,
        kv_ratio=3.0,
        local_range = 9,
        color_fix = True,
        frame_chunk_handler: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        # 只接受 cfg=1.0（与原代码一致）
        assert cfg_scale == 1.0, "cfg_scale must be 1.0"

        # 要求：必须先 init_cross_kv()
        if self.prompt_emb_posi is None or 'context' not in self.prompt_emb_posi:
            raise RuntimeError(
                "Cross-Attn KV 未初始化。请在调用 __call__ 前先执行：\n"
                "    pipe.init_cross_kv()\n"
                "或传入自定义 context：\n"
                "    pipe.init_cross_kv(context_tensor=your_context_tensor)"
            )

        # 尺寸修正
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        # Tiler 参数
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        color_chunk_size = None
        if color_fix:
            color_chunk_size = self._suggest_color_chunk_size(height, width)

        # 初始化噪声
        if if_buffer:
            noise = self.generate_noise((1, 16, (num_frames - 1) // 4, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
        else:
            noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
        # noise = noise.to(dtype=self.torch_dtype, device=self.device)
        latents = noise

        process_total_num = (num_frames - 1) // 8 - 2
        is_stream = True

        # 清理可能存在的 LQ_proj_in cache
        if hasattr(self.dit, "LQ_proj_in"):
            self.dit.LQ_proj_in.clear_cache()

        self.TCDecoder.clean_mem()
        LQ_pre_idx = 0
        LQ_cur_idx = 0
        frames_total = [] if frame_chunk_handler is None else None

        with torch.no_grad():
            # Overlapped two-stage schedule (single-video acceleration)
            use_overlap = (
                self.pp_overlap and self.pp_devices is not None and isinstance(self.pp_devices, (list, tuple)) and len(self.pp_devices) >= 2
            )
            if use_overlap:
                dev0, dev1 = self.pp_devices[0], self.pp_devices[-1]
                # Dedicated copy stream on dev1 so activation copies don't block compute stream
                try:
                    copy_stream_dev1 = torch.cuda.Stream(device=torch.device(dev1))
                except Exception:
                    copy_stream_dev1 = None

                # helpers: run a range of DiT blocks; adapted from model_fn_wan_video
                def _run_blocks_range(x_tokens, context, t_mod_dev, freqs_dev, f, h, w,
                                      local_num, topk, kv_len, is_full_block, is_stream,
                                      block_start, block_end,
                                      pre_cache_k, pre_cache_v,
                                      LQ_latents_for_win):
                    compute_device = x_tokens.device
                    win = (2, 8, 8)
                    seqlen = f // win[0]
                    window_size = win[0] * h * w // 128
                    # loop over specified blocks
                    for block_id in range(block_start, block_end + 1):
                        block = self.dit.blocks[block_id]
                        # device guard
                        try:
                            p = next(block.parameters())
                            if p.device != compute_device:
                                block.to(compute_device)
                        except StopIteration:
                            pass
                        # LQ latent addend
                        if LQ_latents_for_win is not None and block_id < len(LQ_latents_for_win):
                            addend = LQ_latents_for_win[block_id]
                            if addend.device != x_tokens.device:
                                addend = addend.to(x_tokens.device, non_blocking=True)
                            x_tokens = x_tokens + addend
                        # caches
                        def _prepare_cache(cache_list, idx):
                            if cache_list is None:
                                return None
                            tensor = cache_list[idx]
                            if tensor is None:
                                return None
                            if tensor.device == compute_device:
                                return tensor
                            return tensor.to(compute_device, non_blocking=False)
                        def _offload_cache(tensor):
                            if tensor is None or self.cache_offload_device is None:
                                return tensor
                            if tensor.device == self.cache_offload_device:
                                return tensor
                            non_blocking = self.cache_offload_device != "cpu"
                            return tensor.to(self.cache_offload_device, non_blocking=non_blocking)
                        cache_k = _prepare_cache(pre_cache_k, block_id) if pre_cache_k is not None else None
                        cache_v = _prepare_cache(pre_cache_v, block_id) if pre_cache_v is not None else None
                        # block forward
                        x_tokens, last_pre_cache_k, last_pre_cache_v = block(
                            x_tokens, context, t_mod_dev, freqs_dev, f, h, w,
                            local_num, topk,
                            block_id=block_id,
                            kv_len=kv_len,
                            is_full_block=is_full_block,
                            is_stream=is_stream,
                            pre_cache_k=cache_k,
                            pre_cache_v=cache_v,
                            local_range = local_range,
                        )
                        if pre_cache_k is not None: pre_cache_k[block_id] = _offload_cache(last_pre_cache_k)
                        if pre_cache_v is not None: pre_cache_v[block_id] = _offload_cache(last_pre_cache_v)
                    return x_tokens

                # split index
                if self.pp_split_idx is None:
                    split_idx = len(self.dit.blocks) // 2 - 1
                    split_idx = max(0, min(split_idx, len(self.dit.blocks) - 2))
                else:
                    split_idx = self.pp_split_idx

                pre_cache_k = [None] * len(self.dit.blocks)
                pre_cache_v = [None] * len(self.dit.blocks)
                pending = None  # holds tuple for previous window

                for cur_process_idx in tqdm(range(process_total_num)):
                    # Build LQ_latents and current latents (dev0)
                    if cur_process_idx == 0:
                        pre_cache_k = [None] * len(self.dit.blocks)
                        pre_cache_v = [None] * len(self.dit.blocks)
                        LQ_latents = None
                        inner_loop_num = 7
                        for inner_idx in range(inner_loop_num):
                            start_idx = max(0, inner_idx*4-3)
                            end_idx = (inner_idx+1)*4-3
                            cur = self.denoising_model().LQ_proj_in.stream_forward(
                                self._fetch_lq_clip(
                                    LQ_video,
                                    start_idx,
                                    end_idx,
                                    device=self.device,
                                )
                            ) if LQ_video is not None else None
                            if cur is None:
                                continue
                            if LQ_latents is None:
                                LQ_latents = cur
                            else:
                                for layer_idx in range(len(LQ_latents)):
                                    LQ_latents[layer_idx] = torch.cat([LQ_latents[layer_idx], cur[layer_idx]], dim=1)
                        LQ_cur_idx = (inner_loop_num-1)*4-3
                        cur_latents = latents[:, :, :6, :, :]
                    else:
                        LQ_latents = None
                        inner_loop_num = 2
                        for inner_idx in range(inner_loop_num):
                            start_idx = cur_process_idx*8+17+inner_idx*4
                            end_idx = cur_process_idx*8+21+inner_idx*4
                            cur = self.denoising_model().LQ_proj_in.stream_forward(
                                self._fetch_lq_clip(
                                    LQ_video,
                                    start_idx,
                                    end_idx,
                                    device=self.device,
                                )
                            ) if LQ_video is not None else None
                            if cur is None:
                                continue
                            if LQ_latents is None:
                                LQ_latents = cur
                            else:
                                for layer_idx in range(len(LQ_latents)):
                                    LQ_latents[layer_idx] = torch.cat([LQ_latents[layer_idx], cur[layer_idx]], dim=1)
                        LQ_cur_idx = cur_process_idx*8+21+(inner_loop_num-2)*4
                        cur_latents = latents[:, :, 4+cur_process_idx*2:6+cur_process_idx*2, :, :]

                    # patchify and RoPE per-device for this window
                    x_tokens, (f, h, w) = self.dit.patchify(cur_latents)
                    # pre-stage freqs/t_mod per device（使用统一的 3D RoPE 构造，避免维度漂移）
                    head_dim = self.dit.blocks[0].self_attn.head_dim
                    if cur_process_idx == 0:
                        freqs_cpu, self.dit.freqs = build_3d_freqs(
                            getattr(self.dit, "freqs", None),
                            head_dim=head_dim,
                            f=f,
                            h=h,
                            w=w,
                            device="cpu",
                            f_offset=0,
                        )
                    else:
                        freqs_cpu, self.dit.freqs = build_3d_freqs(
                            getattr(self.dit, "freqs", None),
                            head_dim=head_dim,
                            f=f,
                            h=h,
                            w=w,
                            device="cpu",
                            f_offset=4 + cur_process_idx * 2,
                        )
                    freqs_dev0 = freqs_cpu.to(dev0, non_blocking=True)
                    freqs_dev1 = freqs_cpu.to(dev1, non_blocking=True)
                    t_mod_dev0 = self.t_mod.to(dev0, non_blocking=True)
                    t_mod_dev1 = self.t_mod.to(dev1, non_blocking=True)

                    # set scheduling params for this window
                    win = (2, 8, 8)
                    seqlen = f // win[0]
                    local_num = seqlen
                    window_size = win[0] * h * w // 128
                    square_num = window_size * window_size
                    topk = int(square_num * topk_ratio) - 1
                    kv_len = int(kv_ratio)

                    # ensure tokens on dev0
                    if str(x_tokens.device) != str(torch.device(dev0)):
                        x_tokens = x_tokens.to(dev0, non_blocking=True)
                    # If we have a pending window, finish its Stage1 now on dev1 first,
                    # then immediately start Stage0 for the current window on dev0 to overlap.
                    if pending is not None:
                        (prev_idx, prev_x_mid_dev1, prev_LQ_latents, prev_f, prev_h, prev_w,
                         prev_freqs_dev1, prev_t_mod_dev1, prev_cur_latents, prev_LQ_pre_idx, prev_LQ_cur_idx) = pending
                        # Stage1 for previous window
                        x_tail = _run_blocks_range(
                            prev_x_mid_dev1, None, prev_t_mod_dev1, prev_freqs_dev1,
                            prev_f, prev_h, prev_w,
                            local_num, topk, kv_len, is_full_block, True,
                            split_idx+1, len(self.dit.blocks)-1, pre_cache_k, pre_cache_v, prev_LQ_latents)
                        # head + unpatchify on dev1
                        # ensure head on dev1
                        try:
                            if next(self.dit.head.parameters()).device != torch.device(dev1):
                                self.dit.head.to(dev1)
                        except StopIteration:
                            pass
                        noise_pred_posi_prev = self.dit.head(x_tail, self.t.to(device=dev1, non_blocking=True))
                        noise_pred_posi_prev = self.dit.unpatchify(noise_pred_posi_prev, (prev_f, prev_h, prev_w))
                        # Update latent for previous window on GPU1; defer decode to after Stage0 scheduling
                        prev_latents_dev1 = prev_cur_latents.to(dev1, non_blocking=True)
                        prev_latents_dev1 = prev_latents_dev1 - noise_pred_posi_prev

                    # Immediately schedule Stage0 on dev0 for current window to maximize overlap
                    if str(x_tokens.device) != str(torch.device(dev0)):
                        x_tokens = x_tokens.to(dev0, non_blocking=True)
                    x_mid = _run_blocks_range(
                        x_tokens, None, t_mod_dev0, freqs_dev0, f, h, w,
                        local_num, topk, kv_len, is_full_block, True,
                        0, split_idx, pre_cache_k, pre_cache_v, LQ_latents)
                    if copy_stream_dev1 is not None:
                        with torch.cuda.stream(copy_stream_dev1):
                            x_mid_dev1 = x_mid.to(dev1, non_blocking=True)
                    else:
                        x_mid_dev1 = x_mid.to(dev1, non_blocking=True)

                    # Now finish decode/color/cpu-copy for the previous window (on GPU1)
                    if pending is not None:
                        cur_LQ_frame = self._fetch_lq_clip(
                            LQ_video,
                            prev_LQ_pre_idx,
                            prev_LQ_cur_idx,
                            device=dev1,
                        )
                        cur_frames = self.TCDecoder.decode_video(
                            prev_latents_dev1.transpose(1, 2),
                            parallel=False,
                            show_progress_bar=False,
                            cond=cur_LQ_frame,
                        ).transpose(1, 2).mul_(2).sub_(1)
                        try:
                            if color_fix:
                                cur_frames = self.ColorCorrector(
                                    cur_frames,
                                    cur_LQ_frame,
                                    clip_range=(-1, 1),
                                    method='wavelet',
                                    chunk_size=color_chunk_size,
                                )
                        except Exception:
                            pass
                        cpu_chunk = cur_frames.to('cpu')
                        if frame_chunk_handler is None:
                            frames_total.append(cpu_chunk)
                        else:
                            frame_chunk_handler(cpu_chunk)
                        self._release_lq_frames(LQ_video, prev_LQ_cur_idx)
                        LQ_pre_idx = prev_LQ_cur_idx

                    # set new pending window (current) to be finished next iter
                    pending = (cur_process_idx, x_mid_dev1, LQ_latents, f, h, w,
                               freqs_dev1, t_mod_dev1, cur_latents, LQ_pre_idx, LQ_cur_idx)

                # flush tail: finish last pending window Stage1
                if pending is not None:
                    (prev_idx, prev_x_mid_dev1, prev_LQ_latents, prev_f, prev_h, prev_w,
                     prev_freqs_dev1, prev_t_mod_dev1, prev_cur_latents, prev_LQ_pre_idx, prev_LQ_cur_idx) = pending
                    x_tail = _run_blocks_range(
                        prev_x_mid_dev1, None, prev_t_mod_dev1, prev_freqs_dev1,
                        prev_f, prev_h, prev_w,
                        local_num, topk, kv_len, is_full_block, True,
                        split_idx+1, len(self.dit.blocks)-1, pre_cache_k, pre_cache_v, prev_LQ_latents)
                    try:
                        if next(self.dit.head.parameters()).device != torch.device(dev1):
                            self.dit.head.to(dev1)
                    except StopIteration:
                        pass
                    noise_pred_posi_prev = self.dit.head(x_tail, self.t.to(device=dev1, non_blocking=True))
                    noise_pred_posi_prev = self.dit.unpatchify(noise_pred_posi_prev, (prev_f, prev_h, prev_w))
                    # Keep computation on dev1
                    prev_latents_dev1 = prev_cur_latents.to(dev1, non_blocking=True)
                    prev_latents_dev1 = prev_latents_dev1 - noise_pred_posi_prev
                    cur_LQ_frame = self._fetch_lq_clip(
                        LQ_video,
                        prev_LQ_pre_idx,
                        prev_LQ_cur_idx,
                        device=dev1,
                    )
                    cur_frames = self.TCDecoder.decode_video(
                        prev_latents_dev1.transpose(1, 2),
                        parallel=False,
                        show_progress_bar=False,
                        cond=cur_LQ_frame,
                    ).transpose(1, 2).mul_(2).sub_(1)
                    try:
                        if color_fix:
                            cur_frames = self.ColorCorrector(
                                cur_frames,
                                cur_LQ_frame,
                                clip_range=(-1, 1),
                                method='wavelet',
                                chunk_size=color_chunk_size,
                            )
                    except Exception:
                        pass
                    if frame_chunk_handler is None:
                        frames_total.append(cur_frames.to('cpu'))
                    else:
                        frame_chunk_handler(cur_frames.to('cpu'))
                    self._release_lq_frames(LQ_video, prev_LQ_cur_idx)
                # concat frames if needed
                if frame_chunk_handler is not None:
                    return None
                frames_cat = torch.cat(frames_total, dim=2) if len(frames_total) > 0 else None
                return frames_cat[0] if frames_cat is not None else None

            # Non-overlap path (default)
            for cur_process_idx in tqdm(range(process_total_num)):
                if cur_process_idx == 0:
                    pre_cache_k = [None] * len(self.dit.blocks)
                    pre_cache_v = [None] * len(self.dit.blocks)
                    LQ_latents = None
                    inner_loop_num = 7
                    for inner_idx in range(inner_loop_num):
                        start_idx = max(0, inner_idx*4-3)
                        end_idx = (inner_idx+1)*4-3
                        cur = self.denoising_model().LQ_proj_in.stream_forward(
                            self._fetch_lq_clip(
                                LQ_video,
                                start_idx,
                                end_idx,
                                device=self.device,
                            )
                        ) if LQ_video is not None else None
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat([LQ_latents[layer_idx], cur[layer_idx]], dim=1)
                    LQ_cur_idx = (inner_loop_num-1)*4-3
                    cur_latents = latents[:, :, :6, :, :]
                else:
                    LQ_latents = None
                    inner_loop_num = 2
                    for inner_idx in range(inner_loop_num):
                        start_idx = cur_process_idx*8+17+inner_idx*4
                        end_idx = cur_process_idx*8+21+inner_idx*4
                        cur = self.denoising_model().LQ_proj_in.stream_forward(
                            self._fetch_lq_clip(
                                LQ_video,
                                start_idx,
                                end_idx,
                                device=self.device,
                            )
                        ) if LQ_video is not None else None
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat([LQ_latents[layer_idx], cur[layer_idx]], dim=1)
                    LQ_cur_idx = cur_process_idx*8+21+(inner_loop_num-2)*4
                    cur_latents = latents[:, :, 4+cur_process_idx*2:6+cur_process_idx*2, :, :]

                # 推理（无 motion_controller / vace）
                noise_pred_posi, pre_cache_k, pre_cache_v = model_fn_wan_video(
                    self.dit,
                    x=cur_latents,
                    timestep=self.timestep,
                    context=None,
                    tea_cache=None,
                    use_unified_sequence_parallel=False,
                    LQ_latents=LQ_latents,
                    is_full_block=is_full_block,
                    is_stream=is_stream,
                    pre_cache_k=pre_cache_k,
                    pre_cache_v=pre_cache_v,
                    topk_ratio=topk_ratio,
                    kv_ratio=kv_ratio,
                    cur_process_idx=cur_process_idx,
                    t_mod=self.t_mod,
                    t=self.t,
                    local_range = local_range,
                    cache_offload_device=self.cache_offload_device,
                    pp_devices=self.pp_devices,
                    pp_split_idx=self.pp_split_idx,
                )

                # 更新 latent
                cur_latents = cur_latents - noise_pred_posi
                # Decode
                cur_LQ_frame = self._fetch_lq_clip(
                    LQ_video,
                    LQ_pre_idx,
                    LQ_cur_idx,
                    device=self.device,
                )
                cur_frames = self.TCDecoder.decode_video(
                    cur_latents.transpose(1, 2),
                    parallel=False,
                    show_progress_bar=False,
                    cond=cur_LQ_frame,
                ).transpose(1, 2).mul_(2).sub_(1)

                # 颜色校正（wavelet）
                try:
                    if color_fix:
                        cur_frames = self.ColorCorrector(
                            cur_frames.to(device=self.device),
                            cur_LQ_frame,
                            clip_range=(-1, 1),
                            chunk_size=color_chunk_size,
                            method='adain'
                        )
                except:
                    pass

                cpu_chunk = cur_frames.to('cpu')
                if frame_chunk_handler is not None:
                    frame_chunk_handler(cpu_chunk)
                else:
                    frames_total.append(cpu_chunk)
                LQ_pre_idx = LQ_cur_idx
                self._release_lq_frames(LQ_video, LQ_pre_idx)

            if frame_chunk_handler is not None:
                return None
            frames = torch.cat(frames_total, dim=2)

        return frames[0]


# -----------------------------
# TeaCache（保留原逻辑；此处默认不启用）
# -----------------------------
class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B":  [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P":  [8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            should_calc = not (self.accumulated_rel_l1_distance < self.rel_l1_thresh)
            if should_calc:
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step = (self.step + 1) % self.num_inference_steps
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


# -----------------------------
# 简化版模型前向封装（无 vace / 无 motion_controller）
# -----------------------------
def model_fn_wan_video(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    tea_cache: Optional[TeaCache] = None,
    use_unified_sequence_parallel: bool = False,
    LQ_latents: Optional[torch.Tensor] = None,
    is_full_block: bool = False,
    is_stream: bool = False,
    pre_cache_k: Optional[list[torch.Tensor]] = None,
    pre_cache_v: Optional[list[torch.Tensor]] = None,
    topk_ratio: float = 2.0,
    kv_ratio: float = 3.0,
    cur_process_idx: int = 0,
    t_mod : torch.Tensor = None,
    t : torch.Tensor = None,
    local_range: int = 9,
    cache_offload_device: Optional[str] = None,
    pp_devices: Optional[list[str]] = None,
    pp_split_idx: Optional[int] = None,
    **kwargs,
):
    # patchify
    x, (f, h, w) = dit.patchify(x)
    compute_device = x.device
    origin_device = x.device

    def _prepare_cache(cache_list, idx):
        if cache_list is None:
            return None
        tensor = cache_list[idx]
        if tensor is None:
            return None
        if tensor.device == compute_device:
            return tensor
        return tensor.to(compute_device, non_blocking=False)

    def _offload_cache(tensor):
        if tensor is None or cache_offload_device is None:
            return tensor
        if tensor.device == cache_offload_device:
            return tensor
        non_blocking = cache_offload_device != "cpu"
        return tensor.to(cache_offload_device, non_blocking=non_blocking)

    win = (2, 8, 8)
    seqlen = f // win[0]
    local_num = seqlen
    window_size = win[0] * h * w // 128
    square_num = window_size * window_size
    topk = int(square_num * topk_ratio) - 1
    kv_len = int(kv_ratio)

    # RoPE 位置（分段），并预先在两个设备上各保存一份以避免每层重复搬运
    head_dim = dit.blocks[0].self_attn.head_dim
    if cur_process_idx == 0:
        freqs_cpu, dit.freqs = build_3d_freqs(
            getattr(dit, "freqs", None),
            head_dim=head_dim,
            f=f,
            h=h,
            w=w,
            device="cpu",
            f_offset=0,
        )
    else:
        freqs_cpu, dit.freqs = build_3d_freqs(
            getattr(dit, "freqs", None),
            head_dim=head_dim,
            f=f,
            h=h,
            w=w,
            device="cpu",
            f_offset=4 + cur_process_idx * 2,
        )
    # Default single-device
    freqs = freqs_cpu.to(x.device, non_blocking=True)
    # If pp enabled, pre-stage on both devices
    freqs_dev0 = freqs
    freqs_dev1 = None
    t_mod_dev0 = t_mod
    t_mod_dev1 = None
    if pp_devices is not None and isinstance(pp_devices, (list, tuple)) and len(pp_devices) >= 2:
        dev0, dev1 = pp_devices[0], pp_devices[-1]
        if str(torch.device(dev0)) != str(freqs.device):
            freqs_dev0 = freqs_cpu.to(dev0, non_blocking=True)
        freqs_dev1 = freqs_cpu.to(dev1, non_blocking=True) if str(torch.device(dev1)) != str(freqs.device) else freqs
        # pre-stage time modulation
        t_mod_dev0 = t_mod.to(dev0, non_blocking=True) if str(torch.device(dev0)) != str(t_mod.device) else t_mod
        t_mod_dev1 = t_mod.to(dev1, non_blocking=True) if str(torch.device(dev1)) != str(t_mod.device) else t_mod

    # TeaCache（默认不启用）
    tea_cache_update = tea_cache.check(dit, x, t_mod) if tea_cache is not None else False

    # 统一序列并行（此处默认关闭）
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                             get_sequence_parallel_world_size,
                                             get_sp_group)
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    # Pipeline-parallel planning (optional)
    pp_enabled = pp_devices is not None and isinstance(pp_devices, (list, tuple)) and len(pp_devices) >= 2
    if pp_enabled:
        dev0, dev1 = pp_devices[0], pp_devices[-1]
        if pp_split_idx is None:
            pp_split_idx = len(dit.blocks) // 2 - 1
            pp_split_idx = max(0, min(pp_split_idx, len(dit.blocks) - 2))

    # Block 堆叠
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        for block_id, block in enumerate(dit.blocks):
            # Select device for this block if pp is enabled
            if pp_enabled:
                target_device = dev0 if block_id <= pp_split_idx else dev1
                if str(x.device) != str(torch.device(target_device)):
                    x = x.to(target_device, non_blocking=False)
                # keep caches / helper tensors on the same device
                compute_device = x.device
                # Select pre-staged tensors for this device
                freqs_cur = freqs_dev0 if str(compute_device) == str(torch.device(dev0)) else (freqs_dev1 if freqs_dev1 is not None else freqs.to(compute_device, non_blocking=True))
                t_mod_cur = t_mod_dev0 if str(compute_device) == str(torch.device(dev0)) else (t_mod_dev1 if t_mod_dev1 is not None else t_mod.to(compute_device, non_blocking=True))
                # ensure block is on the right device (fallback safety)
                try:
                    p = next(block.parameters())
                    if p.device != compute_device:
                        block.to(compute_device)
                except StopIteration:
                    pass
            else:
                freqs_cur = freqs
                t_mod_cur = t_mod
            if LQ_latents is not None and block_id < len(LQ_latents):
                addend = LQ_latents[block_id]
                if pp_enabled and addend.device != x.device:
                    addend = addend.to(x.device, non_blocking=False)
                x = x + addend
            cache_k = _prepare_cache(pre_cache_k, block_id) if pre_cache_k is not None else None
            cache_v = _prepare_cache(pre_cache_v, block_id) if pre_cache_v is not None else None
            x, last_pre_cache_k, last_pre_cache_v = block(
                x, context, t_mod_cur, freqs_cur, f, h, w,
                local_num, topk,
                block_id=block_id,
                kv_len=kv_len,
                is_full_block=is_full_block,
                is_stream=is_stream,
                pre_cache_k=cache_k,
                pre_cache_v=cache_v,
                local_range = local_range,
            )
            if pre_cache_k is not None: pre_cache_k[block_id] = _offload_cache(last_pre_cache_k)
            if pre_cache_v is not None: pre_cache_v[block_id] = _offload_cache(last_pre_cache_v)

    # Final head on the device of last stage when pp is enabled
    if pp_enabled:
        last_device = dev1
        if str(x.device) != str(torch.device(last_device)):
            x = x.to(last_device, non_blocking=False)
        try:
            if next(dit.head.parameters()).device != x.device:
                dit.head.to(x.device)
        except StopIteration:
            pass
        t = t.to(x.device, non_blocking=False)
        x = dit.head(x, t)
    else:
        x = dit.head(x, t)
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import get_sp_group
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    x = dit.unpatchify(x, (f, h, w))
    # Move result back to the origin device so caller tensors match
    if pp_enabled and str(x.device) != str(origin_device):
        x = x.to(origin_device, non_blocking=False)
    return x, pre_cache_k, pre_cache_v
