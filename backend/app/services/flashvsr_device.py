"""FlashVSR 设备与多卡配置工具."""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch

from app.config import settings


def resolve_device() -> str:
    """根据配置和可用性决定默认设备."""
    override = (settings.FLASHVSR_DEVICE or "").strip()
    if override:
        if override.startswith("cuda"):
            if torch.cuda.is_available():
                try:
                    if ":" in override:
                        idx = int(override.split(":", 1)[1])
                        torch.cuda.set_device(idx)
                except Exception:
                    pass
                return override
            return "cpu"
        if override == "cpu":
            return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def decide_cache_offload_device(device: str) -> Tuple[Optional[str], Optional[str]]:
    """
    根据 FLASHVSR_CACHE_OFFLOAD 决定 KV cache 是否下放到 CPU，返回 (device, reason)。
    """
    mode = (settings.FLASHVSR_CACHE_OFFLOAD or "auto").strip().lower()
    allowed = {"auto", "cpu", "none", "off", "disable"}
    if mode not in allowed:
        raise ValueError(
            f"无效的 FLASHVSR_CACHE_OFFLOAD 配置: {settings.FLASHVSR_CACHE_OFFLOAD}. "
            f"可选值: {', '.join(sorted(allowed))}"
        )
    if not device.startswith("cuda"):
        return None, None

    gpu_index = 0
    try:
        if ":" in device:
            gpu_index = int(device.split(":", 1)[1])
    except Exception:
        gpu_index = 0
    total_gb = torch.cuda.get_device_properties(gpu_index).total_memory / (1024 ** 3)
    threshold = settings.FLASHVSR_CACHE_OFFLOAD_AUTO_THRESHOLD_GB

    if mode == "cpu":
        return "cpu", "forced via FLASHVSR_CACHE_OFFLOAD=cpu"
    if mode == "auto" and total_gb <= threshold:
        return "cpu", f"auto: GPU {total_gb:.1f} GB ≤ {threshold:.1f} GB"
    return None, None


def parse_pipeline_parallel() -> Tuple[Optional[List[str]], Optional[int]]:
    """
    解析流水线并行配置, 返回 (devices, split_index)；若未启用则返回 (None, None)。
    """
    raw = (settings.FLASHVSR_PP_DEVICES or "").strip()
    if not raw:
        return None, None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    devices: List[str] = []
    for p in parts:
        if p.startswith("cuda"):
            devices.append(p)
        elif p.isdigit():
            devices.append(f"cuda:{p}")
        else:
            devices.append(p)
    if len(devices) < 2:
        return None, None

    split_raw = (settings.FLASHVSR_PP_SPLIT_BLOCK or "auto").strip().lower()
    if split_raw in ("", "auto"):
        split_index: Optional[int] = None
    else:
        try:
            split_index = int(split_raw)
        except Exception:
            split_index = None
    return devices, split_index

