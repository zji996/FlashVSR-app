from __future__ import annotations

from typing import Optional

import torch

from ..models.wan_video_dit import WanModel, build_3d_freqs


def model_fn_wan_video(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: Optional[torch.Tensor],
    LQ_latents: Optional[torch.Tensor] = None,
    is_full_block: bool = False,
    is_stream: bool = False,
    pre_cache_k: Optional[list[torch.Tensor]] = None,
    pre_cache_v: Optional[list[torch.Tensor]] = None,
    topk_ratio: float = 2.0,
    kv_ratio: float = 3.0,
    cur_process_idx: int = 0,
    t_mod: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    local_range: int = 9,
    cache_offload_device: Optional[str] = None,
    pp_devices: Optional[list[str]] = None,
    pp_split_idx: Optional[int] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], Optional[list[torch.Tensor]]]:
    """
    简化版 WanVideo DiT 前向封装（无 motion_controller / vace）。
    """
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
        freqs_dev1 = (
            freqs_cpu.to(dev1, non_blocking=True)
            if str(torch.device(dev1)) != str(freqs.device)
            else freqs
        )
        # pre-stage time modulation
        if t_mod is not None:
            t_mod_dev0 = (
                t_mod.to(dev0, non_blocking=True)
                if str(torch.device(dev0)) != str(t_mod.device)
                else t_mod
            )
            t_mod_dev1 = (
                t_mod.to(dev1, non_blocking=True)
                if str(torch.device(dev1)) != str(t_mod.device)
                else t_mod
            )

    # Pipeline-parallel planning (optional)
    pp_enabled = (
        pp_devices is not None and isinstance(pp_devices, (list, tuple)) and len(pp_devices) >= 2
    )
    if pp_enabled:
        dev0, dev1 = pp_devices[0], pp_devices[-1]
        if pp_split_idx is None:
            pp_split_idx = len(dit.blocks) // 2 - 1
            pp_split_idx = max(0, min(pp_split_idx, len(dit.blocks) - 2))

    # Block 堆叠
    for block_id, block in enumerate(dit.blocks):
        # Select device for this block if pp is enabled
        if pp_enabled:
            target_device = dev0 if block_id <= pp_split_idx else dev1
            if str(x.device) != str(torch.device(target_device)):
                x = x.to(target_device, non_blocking=False)
            # keep caches / helper tensors on the same device
            compute_device = x.device
            # Select pre-staged tensors for this device
            freqs_cur = (
                freqs_dev0
                if str(compute_device) == str(torch.device(dev0))
                else (
                    freqs_dev1
                    if freqs_dev1 is not None
                    else freqs.to(compute_device, non_blocking=True)
                )
            )
            t_mod_cur = (
                t_mod_dev0
                if str(compute_device) == str(torch.device(dev0))
                else (
                    t_mod_dev1
                    if t_mod_dev1 is not None
                    else t_mod.to(compute_device, non_blocking=True)
                )
            )
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
            x,
            context,
            t_mod_cur,
            freqs_cur,
            f,
            h,
            w,
            local_num,
            topk,
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
        if t is not None:
            t = t.to(x.device, non_blocking=False)
        x = dit.head(x, t)
    else:
        x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    # Move result back to the origin device so caller tensors match
    if pp_enabled and str(x.device) != str(origin_device):
        x = x.to(origin_device, non_blocking=False)
    return x, pre_cache_k, pre_cache_v
