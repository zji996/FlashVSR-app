from pathlib import Path
import sys

import torch

# Ensure backend/ is on sys.path so `import app` works when running tests via uv.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services import flashvsr_io
from app.config import settings


def test_largest_8n1_leq_basic():
    helper = flashvsr_io.largest_8n1_leq

    assert helper(0) == 0
    assert helper(1) == 1
    assert helper(2) == 1
    assert helper(8) == 1
    assert helper(9) == 9
    assert helper(15) == 9
    assert helper(16) == 9
    assert helper(17) == 17


def test_compute_scaled_dims_alignment():
    scale = 2.0
    base_width = 1920
    base_height = 1080

    s_width, s_height, t_width, t_height = flashvsr_io.compute_scaled_dims(
        base_width, base_height, scale
    )

    assert s_width == int(round(base_width * scale))
    assert s_height == int(round(base_height * scale))
    assert t_width % 128 == 0
    assert t_height % 128 == 0
    assert t_width <= s_width
    assert t_height <= s_height


def test_estimate_video_bytes_matches_dtype():
    total_frames = 10
    height = 720
    width = 1280

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        element_bytes = torch.finfo(dtype).bits // 8
        expected = total_frames * height * width * 3 * element_bytes
        computed = flashvsr_io.estimate_video_bytes(
            total_frames, height, width, dtype
        )
        assert computed == expected


def test_should_stream_video_respects_prefetch(monkeypatch):
    # prefetch > 0 -> streaming enabled
    monkeypatch.setattr(settings, "FLASHVSR_STREAMING_PREFETCH_FRAMES", 16, raising=False)
    assert flashvsr_io.should_stream_video(100, 720, 1280, torch.bfloat16) is True

    # prefetch == 0 -> streaming disabled
    monkeypatch.setattr(settings, "FLASHVSR_STREAMING_PREFETCH_FRAMES", 0, raising=False)
    assert flashvsr_io.should_stream_video(100, 720, 1280, torch.bfloat16) is False


def test_streaming_buffer_auto_uses_min_working_set(monkeypatch):
    height = 1024
    width = 1920
    dtype = torch.bfloat16

    # 使用默认预读帧数 25，自动模式（LQ 上限未设置或为 0）。
    monkeypatch.setattr(settings, "FLASHVSR_STREAMING_PREFETCH_FRAMES", 25, raising=False)
    monkeypatch.setattr(settings, "FLASHVSR_STREAMING_LQ_MAX_BYTES", 0, raising=False)

    capacity_frames, prefetch, per_frame_bytes, limit_bytes = flashvsr_io._compute_streaming_buffer_config(
        total_frames=100,
        height=height,
        width=width,
        dtype=dtype,
    )

    expected_per_frame = flashvsr_io.estimate_video_bytes(1, height, width, dtype)
    assert per_frame_bytes == expected_per_frame
    # 目标工作集不少于 50 帧。
    assert prefetch == 25
    assert capacity_frames == 50
    assert limit_bytes == expected_per_frame * 50


def test_streaming_buffer_ignores_too_small_config_limit(monkeypatch):
    height = 1024
    width = 1920
    dtype = torch.bfloat16

    monkeypatch.setattr(settings, "FLASHVSR_STREAMING_PREFETCH_FRAMES", 25, raising=False)

    per_frame = flashvsr_io.estimate_video_bytes(1, height, width, dtype)
    auto_limit = per_frame * 50
    # 配置一个明显偏小的上限，函数应自动提升到安全值。
    small_limit = per_frame * 10
    monkeypatch.setattr(
        settings,
        "FLASHVSR_STREAMING_LQ_MAX_BYTES",
        small_limit,
        raising=False,
    )

    capacity_frames, prefetch, _, limit_bytes = flashvsr_io._compute_streaming_buffer_config(
        total_frames=100,
        height=height,
        width=width,
        dtype=dtype,
    )

    assert prefetch == 25
    assert capacity_frames == 50
    assert limit_bytes == auto_limit
