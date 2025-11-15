from typing import Optional, Callable

import torch
from tqdm import tqdm

from ..models import ModelManager
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d, build_3d_freqs
from ..models.wan_video_vae import WanVideoVAE
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from .flashvsr_color import TorchColorCorrectorWavelet
from .wan_video_runtime import model_fn_wan_video


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
        self.prompt_emb_posi = None
        self.ColorCorrector = TorchColorCorrectorWavelet(levels=5)

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
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = FlashVSRTinyLongPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
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
        LQ_video=None,
        is_full_block=False,
        if_buffer: bool = False,
        topk_ratio=2.0,
        kv_ratio=3.0,
        local_range=9,
        color_fix=True,
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
        color_chunk_size = None
        if color_fix:
            color_chunk_size = self._suggest_color_chunk_size(height, width)

        # 初始化噪声：按窗口生成，避免为整个长视频一次性分配巨大的 latent 张量
        def _make_latents_window(step_index: int) -> torch.Tensor:
            """
            为当前窗口生成噪声 latent。

            原实现使用一个形状为 (1, 16, (num_frames - 1)//4, H/8, W/8) 的全局 latent，
            然后在每个窗口中切片：
              - 第 0 个窗口使用前 6 帧；
              - 之后每个窗口使用 2 帧，并依次向后滑动。
            这会导致在超长视频上分配 O(T) 级别显存。

            这里改为按窗口即时采样：
              - 第 0 个窗口生成 6 帧 latent；
              - 之后每个窗口生成 2 帧 latent；
            这样最大显存占用只与分辨率相关，而与视频总帧数无关。
            """
            if step_index == 0:
                frames = 6
            else:
                frames = 2

            if seed is None:
                window_seed = None
            else:
                # 保持可复现：不同窗口使用稳定偏移的种子
                window_seed = seed + step_index

            return self.generate_noise(
                (1, 16, frames, height // 8, width // 8),
                seed=window_seed,
                device=self.device,
                dtype=self.torch_dtype,
            )

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
            # Aggressive mode：在 basic overlap 上进一步拆分前处理阶段
            overlap_mode = getattr(self, "pp_overlap_mode", "basic")
            overlap_mode = (overlap_mode or "basic").lower()

            if use_overlap and overlap_mode == "aggressive":
                dev0, dev1 = self.pp_devices[0], self.pp_devices[-1]

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
                    for block_id in range(block_start, block_end + 1):
                        block = self.dit.blocks[block_id]
                        try:
                            p = next(block.parameters())
                            if p.device != compute_device:
                                block.to(compute_device)
                        except StopIteration:
                            pass
                        if LQ_latents_for_win is not None and block_id < len(LQ_latents_for_win):
                            addend = LQ_latents_for_win[block_id]
                            if addend.device != x_tokens.device:
                                addend = addend.to(x_tokens.device, non_blocking=True)
                            x_tokens = x_tokens + addend

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
                        x_tokens, last_pre_cache_k, last_pre_cache_v = block(
                            x_tokens, context, t_mod_dev, freqs_dev, f, h, w,
                            local_num, topk,
                            block_id=block_id,
                            kv_len=kv_len,
                            is_full_block=is_full_block,
                            is_stream=is_stream,
                            pre_cache_k=cache_k,
                            pre_cache_v=cache_v,
                            local_range=local_range,
                        )
                        if pre_cache_k is not None:
                            pre_cache_k[block_id] = _offload_cache(last_pre_cache_k)
                        if pre_cache_v is not None:
                            pre_cache_v[block_id] = _offload_cache(last_pre_cache_v)
                    return x_tokens

                if self.pp_split_idx is None:
                    split_idx = len(self.dit.blocks) // 2 - 1
                    split_idx = max(0, min(split_idx, len(self.dit.blocks) - 2))
                else:
                    split_idx = self.pp_split_idx

                pre_cache_k = [None] * len(self.dit.blocks)
                pre_cache_v = [None] * len(self.dit.blocks)

                class WindowCtx:
                    __slots__ = (
                        "index",
                        "cur_latents",
                        "x_tokens",
                        "LQ_latents",
                        "f",
                        "h",
                        "w",
                        "freqs_cpu",
                        "LQ_pre_idx",
                        "LQ_cur_idx",
                        "x_mid_dev1",
                    )

                    def __init__(
                        self,
                        index,
                        cur_latents,
                        x_tokens,
                        LQ_latents,
                        f,
                        h,
                        w,
                        freqs_cpu,
                        LQ_pre_idx,
                        LQ_cur_idx,
                    ):
                        self.index = index
                        self.cur_latents = cur_latents
                        self.x_tokens = x_tokens
                        self.LQ_latents = LQ_latents
                        self.f = f
                        self.h = h
                        self.w = w
                        self.freqs_cpu = freqs_cpu
                        self.LQ_pre_idx = LQ_pre_idx
                        self.LQ_cur_idx = LQ_cur_idx
                        self.x_mid_dev1 = None

                def _stage_a_prepare(idx, prev_ctx):
                    nonlocal LQ_pre_idx, LQ_cur_idx
                    if idx == 0:
                        local_pre_idx = 0
                        LQ_latents_local = None
                        inner_loop_num_local = 7
                        for inner_idx in range(inner_loop_num_local):
                            start_idx = max(0, inner_idx * 4 - 3)
                            end_idx = (inner_idx + 1) * 4 - 3
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
                            if LQ_latents_local is None:
                                LQ_latents_local = cur
                            else:
                                for layer_idx in range(len(LQ_latents_local)):
                                    LQ_latents_local[layer_idx] = torch.cat(
                                        [LQ_latents_local[layer_idx], cur[layer_idx]], dim=1
                                    )
                        local_cur_idx = (inner_loop_num_local - 1) * 4 - 3
                        cur_latents_local = _make_latents_window(idx)
                    else:
                        if prev_ctx is not None:
                            LQ_pre_idx = prev_ctx.LQ_cur_idx
                        LQ_latents_local = None
                        inner_loop_num_local = 2
                        for inner_idx in range(inner_loop_num_local):
                            start_idx = idx * 8 + 17 + inner_idx * 4
                            end_idx = idx * 8 + 21 + inner_idx * 4
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
                            if LQ_latents_local is None:
                                LQ_latents_local = cur
                            else:
                                for layer_idx in range(len(LQ_latents_local)):
                                    LQ_latents_local[layer_idx] = torch.cat(
                                        [LQ_latents_local[layer_idx], cur[layer_idx]], dim=1
                                    )
                        local_cur_idx = idx * 8 + 21 + (inner_loop_num_local - 2) * 4
                        cur_latents_local = _make_latents_window(idx)

                    x_tokens_local, (f_local, h_local, w_local) = self.dit.patchify(cur_latents_local)
                    head_dim = self.dit.blocks[0].self_attn.head_dim
                    if idx == 0:
                        freqs_cpu_local, self.dit.freqs = build_3d_freqs(
                            getattr(self.dit, "freqs", None),
                            head_dim=head_dim,
                            f=f_local,
                            h=h_local,
                            w=w_local,
                            device="cpu",
                            f_offset=0,
                        )
                    else:
                        freqs_cpu_local, self.dit.freqs = build_3d_freqs(
                            getattr(self.dit, "freqs", None),
                            head_dim=head_dim,
                            f=f_local,
                            h=h_local,
                            w=w_local,
                            device="cpu",
                            f_offset=4 + idx * 2,
                        )
                    if str(x_tokens_local.device) != str(torch.device(dev0)):
                        x_tokens_local = x_tokens_local.to(dev0, non_blocking=True)
                    ctx = WindowCtx(
                        index=idx,
                        cur_latents=cur_latents_local,
                        x_tokens=x_tokens_local,
                        LQ_latents=LQ_latents_local,
                        f=f_local,
                        h=h_local,
                        w=w_local,
                        freqs_cpu=freqs_cpu_local,
                        LQ_pre_idx=LQ_pre_idx,
                        LQ_cur_idx=local_cur_idx,
                    )
                    return ctx

                def _run_stage0_and_copy(ctx: WindowCtx):
                    f_local, h_local, w_local = ctx.f, ctx.h, ctx.w
                    win = (2, 8, 8)
                    seqlen = f_local // win[0]
                    local_num_local = seqlen
                    window_size = win[0] * h_local * w_local // 128
                    square_num = window_size * window_size
                    topk_local = int(square_num * topk_ratio) - 1
                    kv_len_local = int(kv_ratio)

                    freqs_dev0 = ctx.freqs_cpu.to(dev0, non_blocking=True)
                    freqs_dev1 = ctx.freqs_cpu.to(dev1, non_blocking=True)
                    t_mod_dev0 = self.t_mod.to(dev0, non_blocking=True)
                    t_mod_dev1 = self.t_mod.to(dev1, non_blocking=True)

                    x_tokens_local = ctx.x_tokens
                    if x_tokens_local is None:
                        x_tokens_local, _ = self.dit.patchify(ctx.cur_latents)
                        ctx.x_tokens = x_tokens_local
                    if str(x_tokens_local.device) != str(torch.device(dev0)):
                        x_tokens_local = x_tokens_local.to(dev0, non_blocking=True)

                    x_mid = _run_blocks_range(
                        x_tokens_local,
                        None,
                        t_mod_dev0,
                        freqs_dev0,
                        f_local,
                        h_local,
                        w_local,
                        local_num_local,
                        topk_local,
                        kv_len_local,
                        is_full_block,
                        True,
                        0,
                        split_idx,
                        pre_cache_k,
                        pre_cache_v,
                        ctx.LQ_latents,
                    )
                    if copy_stream_dev1 is not None:
                        with torch.cuda.stream(copy_stream_dev1):
                            x_mid_dev1 = x_mid.to(dev1, non_blocking=True)
                    else:
                        x_mid_dev1 = x_mid.to(dev1, non_blocking=True)
                    ctx.freqs_cpu = freqs_dev1
                    ctx.x_mid_dev1 = x_mid_dev1
                    return ctx

                def _run_stage1_and_decode(ctx: WindowCtx):
                    if ctx is None or ctx.x_mid_dev1 is None:
                        return
                    f_local, h_local, w_local = ctx.f, ctx.h, ctx.w
                    win = (2, 8, 8)
                    seqlen = f_local // win[0]
                    local_num_local = seqlen
                    window_size = win[0] * h_local * w_local // 128
                    square_num = window_size * window_size
                    topk_local = int(square_num * topk_ratio) - 1
                    kv_len_local = int(kv_ratio)

                    freqs_dev1 = ctx.freqs_cpu.to(dev1, non_blocking=True)
                    t_mod_dev1 = self.t_mod.to(dev1, non_blocking=True)

                    x_tail = _run_blocks_range(
                        ctx.x_mid_dev1,
                        None,
                        t_mod_dev1,
                        freqs_dev1,
                        f_local,
                        h_local,
                        w_local,
                        local_num_local,
                        topk_local,
                        kv_len_local,
                        is_full_block,
                        True,
                        split_idx + 1,
                        len(self.dit.blocks) - 1,
                        pre_cache_k,
                        pre_cache_v,
                        ctx.LQ_latents,
                    )
                    try:
                        if next(self.dit.head.parameters()).device != torch.device(dev1):
                            self.dit.head.to(dev1)
                    except StopIteration:
                        pass
                    noise_pred_posi = self.dit.head(
                        x_tail, self.t.to(device=dev1, non_blocking=True)
                    )
                    noise_pred_posi = self.dit.unpatchify(noise_pred_posi, (f_local, h_local, w_local))
                    latents_dev1 = ctx.cur_latents.to(dev1, non_blocking=True)
                    latents_dev1 = latents_dev1 - noise_pred_posi
                    cur_LQ_frame = self._fetch_lq_clip(
                        LQ_video,
                        ctx.LQ_pre_idx,
                        ctx.LQ_cur_idx,
                        device=dev1,
                    )
                    cur_frames = self.TCDecoder.decode_video(
                        latents_dev1.transpose(1, 2),
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
                                method="wavelet",
                                chunk_size=color_chunk_size,
                            )
                    except Exception:
                        pass
                    cpu_chunk = cur_frames.to("cpu")
                    if frame_chunk_handler is None:
                        frames_total.append(cpu_chunk)
                    else:
                        frame_chunk_handler(cpu_chunk)
                    self._release_lq_frames(LQ_video, ctx.LQ_cur_idx)

                prev_ctx = None
                cur_ctx = _stage_a_prepare(0, None) if process_total_num > 0 else None

                for cur_process_idx in tqdm(range(process_total_num)):
                    if cur_process_idx > 0 and prev_ctx is not None:
                        _run_stage1_and_decode(prev_ctx)
                        # 释放上一窗口中不再需要的大张量引用
                        prev_ctx.cur_latents = None
                        prev_ctx.x_tokens = None
                        prev_ctx.LQ_latents = None

                    if cur_ctx is None:
                        break

                    cur_ctx = _run_stage0_and_copy(cur_ctx)

                    next_ctx = (
                        _stage_a_prepare(cur_process_idx + 1, cur_ctx)
                        if cur_process_idx + 1 < process_total_num
                        else None
                    )

                    prev_ctx, cur_ctx = cur_ctx, next_ctx

                if prev_ctx is not None:
                    _run_stage1_and_decode(prev_ctx)

                if frame_chunk_handler is not None:
                    return None
                frames_cat = torch.cat(frames_total, dim=2) if len(frames_total) > 0 else None
                return frames_cat[0] if frames_cat is not None else None

            # basic overlap（沿用原有实现）
            if use_overlap and overlap_mode != "aggressive":
                dev0, dev1 = self.pp_devices[0], self.pp_devices[-1]
                try:
                    copy_stream_dev1 = torch.cuda.Stream(device=torch.device(dev1))
                except Exception:
                    copy_stream_dev1 = None

                def _run_blocks_range(x_tokens, context, t_mod_dev, freqs_dev, f, h, w,
                                      local_num, topk, kv_len, is_full_block, is_stream,
                                      block_start, block_end,
                                      pre_cache_k, pre_cache_v,
                                      LQ_latents_for_win):
                    compute_device = x_tokens.device
                    win = (2, 8, 8)
                    seqlen = f // win[0]
                    window_size = win[0] * h * w // 128
                    for block_id in range(block_start, block_end + 1):
                        block = self.dit.blocks[block_id]
                        try:
                            p = next(block.parameters())
                            if p.device != compute_device:
                                block.to(compute_device)
                        except StopIteration:
                            pass
                        if LQ_latents_for_win is not None and block_id < len(LQ_latents_for_win):
                            addend = LQ_latents_for_win[block_id]
                            if addend.device != x_tokens.device:
                                addend = addend.to(x_tokens.device, non_blocking=True)
                            x_tokens = x_tokens + addend

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
                        x_tokens, last_pre_cache_k, last_pre_cache_v = block(
                            x_tokens, context, t_mod_dev, freqs_dev, f, h, w,
                            local_num, topk,
                            block_id=block_id,
                            kv_len=kv_len,
                            is_full_block=is_full_block,
                            is_stream=is_stream,
                            pre_cache_k=cache_k,
                            pre_cache_v=cache_v,
                            local_range=local_range,
                        )
                        if pre_cache_k is not None:
                            pre_cache_k[block_id] = _offload_cache(last_pre_cache_k)
                        if pre_cache_v is not None:
                            pre_cache_v[block_id] = _offload_cache(last_pre_cache_v)
                    return x_tokens

                if self.pp_split_idx is None:
                    split_idx = len(self.dit.blocks) // 2 - 1
                    split_idx = max(0, min(split_idx, len(self.dit.blocks) - 2))
                else:
                    split_idx = self.pp_split_idx

                pre_cache_k = [None] * len(self.dit.blocks)
                pre_cache_v = [None] * len(self.dit.blocks)
                pending = None

                for cur_process_idx in tqdm(range(process_total_num)):
                    if cur_process_idx == 0:
                        pre_cache_k = [None] * len(self.dit.blocks)
                        pre_cache_v = [None] * len(self.dit.blocks)
                        LQ_latents = None
                        inner_loop_num = 7
                        for inner_idx in range(inner_loop_num):
                            start_idx = max(0, inner_idx * 4 - 3)
                            end_idx = (inner_idx + 1) * 4 - 3
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
                                    LQ_latents[layer_idx] = torch.cat(
                                        [LQ_latents[layer_idx], cur[layer_idx]], dim=1
                                    )
                        LQ_cur_idx_local = (inner_loop_num - 1) * 4 - 3
                        cur_latents = _make_latents_window(cur_process_idx)
                    else:
                        LQ_latents = None
                        inner_loop_num = 2
                        for inner_idx in range(inner_loop_num):
                            start_idx = cur_process_idx * 8 + 17 + inner_idx * 4
                            end_idx = cur_process_idx * 8 + 21 + inner_idx * 4
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
                                    LQ_latents[layer_idx] = torch.cat(
                                        [LQ_latents[layer_idx], cur[layer_idx]], dim=1
                                    )
                        LQ_cur_idx_local = cur_process_idx * 8 + 21 + (inner_loop_num - 2) * 4
                        cur_latents = _make_latents_window(cur_process_idx)

                    x_tokens, (f, h, w) = self.dit.patchify(cur_latents)
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

                    win = (2, 8, 8)
                    seqlen = f // win[0]
                    local_num = seqlen
                    window_size = win[0] * h * w // 128
                    square_num = window_size * window_size
                    topk = int(square_num * topk_ratio) - 1
                    kv_len = int(kv_ratio)

                    if str(x_tokens.device) != str(torch.device(dev0)):
                        x_tokens = x_tokens.to(dev0, non_blocking=True)
                    if pending is not None:
                        (
                            prev_idx,
                            prev_x_mid_dev1,
                            prev_LQ_latents,
                            prev_f,
                            prev_h,
                            prev_w,
                            prev_freqs_dev1,
                            prev_t_mod_dev1,
                            prev_cur_latents,
                            prev_LQ_pre_idx,
                            prev_LQ_cur_idx,
                        ) = pending
                        x_tail = _run_blocks_range(
                            prev_x_mid_dev1,
                            None,
                            prev_t_mod_dev1,
                            prev_freqs_dev1,
                            prev_f,
                            prev_h,
                            prev_w,
                            local_num,
                            topk,
                            kv_len,
                            is_full_block,
                            True,
                            split_idx + 1,
                            len(self.dit.blocks) - 1,
                            pre_cache_k,
                            pre_cache_v,
                            prev_LQ_latents,
                        )
                        try:
                            if next(self.dit.head.parameters()).device != torch.device(dev1):
                                self.dit.head.to(dev1)
                        except StopIteration:
                            pass
                        noise_pred_posi_prev = self.dit.head(
                            x_tail, self.t.to(device=dev1, non_blocking=True)
                        )
                        noise_pred_posi_prev = self.dit.unpatchify(
                            noise_pred_posi_prev, (prev_f, prev_h, prev_w)
                        )
                        prev_latents_dev1 = prev_cur_latents.to(dev1, non_blocking=True)
                        prev_latents_dev1 = prev_latents_dev1 - noise_pred_posi_prev

                    if str(x_tokens.device) != str(torch.device(dev0)):
                        x_tokens = x_tokens.to(dev0, non_blocking=True)
                    x_mid = _run_blocks_range(
                        x_tokens,
                        None,
                        t_mod_dev0,
                        freqs_dev0,
                        f,
                        h,
                        w,
                        local_num,
                        topk,
                        kv_len,
                        is_full_block,
                        True,
                        0,
                        split_idx,
                        pre_cache_k,
                        pre_cache_v,
                        LQ_latents,
                    )
                    if copy_stream_dev1 is not None:
                        with torch.cuda.stream(copy_stream_dev1):
                            x_mid_dev1 = x_mid.to(dev1, non_blocking=True)
                    else:
                        x_mid_dev1 = x_mid.to(dev1, non_blocking=True)

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
                                    method="wavelet",
                                    chunk_size=color_chunk_size,
                                )
                        except Exception:
                            pass
                        cpu_chunk = cur_frames.to("cpu")
                        if frame_chunk_handler is None:
                            frames_total.append(cpu_chunk)
                        else:
                            frame_chunk_handler(cpu_chunk)
                        self._release_lq_frames(LQ_video, prev_LQ_cur_idx)
                        LQ_pre_idx = prev_LQ_cur_idx

                    pending = (
                        cur_process_idx,
                        x_mid_dev1,
                        LQ_latents,
                        f,
                        h,
                        w,
                        freqs_dev1,
                        t_mod_dev1,
                        cur_latents,
                        LQ_pre_idx,
                        LQ_cur_idx_local,
                    )

                if pending is not None:
                    (
                        prev_idx,
                        prev_x_mid_dev1,
                        prev_LQ_latents,
                        prev_f,
                        prev_h,
                        prev_w,
                        prev_freqs_dev1,
                        prev_t_mod_dev1,
                        prev_cur_latents,
                        prev_LQ_pre_idx,
                        prev_LQ_cur_idx,
                    ) = pending
                    x_tail = _run_blocks_range(
                        prev_x_mid_dev1,
                        None,
                        prev_t_mod_dev1,
                        prev_freqs_dev1,
                        prev_f,
                        prev_h,
                        prev_w,
                        local_num,
                        topk,
                        kv_len,
                        is_full_block,
                        True,
                        split_idx + 1,
                        len(self.dit.blocks) - 1,
                        pre_cache_k,
                        pre_cache_v,
                        prev_LQ_latents,
                    )
                    try:
                        if next(self.dit.head.parameters()).device != torch.device(dev1):
                            self.dit.head.to(dev1)
                    except StopIteration:
                        pass
                    noise_pred_posi_prev = self.dit.head(
                        x_tail, self.t.to(device=dev1, non_blocking=True)
                    )
                    noise_pred_posi_prev = self.dit.unpatchify(
                        noise_pred_posi_prev, (prev_f, prev_h, prev_w)
                    )
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
                                method="wavelet",
                                chunk_size=color_chunk_size,
                            )
                    except Exception:
                        pass
                    if frame_chunk_handler is None:
                        frames_total.append(cur_frames.to("cpu"))
                    else:
                        frame_chunk_handler(cur_frames.to("cpu"))
                    self._release_lq_frames(LQ_video, prev_LQ_cur_idx)

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
                    cur_latents = _make_latents_window(cur_process_idx)
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
                    cur_latents = _make_latents_window(cur_process_idx)

                # 推理（无 motion_controller / vace）
                noise_pred_posi, pre_cache_k, pre_cache_v = model_fn_wan_video(
                    self.dit,
                    x=cur_latents,
                    timestep=self.timestep,
                    context=None,
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
