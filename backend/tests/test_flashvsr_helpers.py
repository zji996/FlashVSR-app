from pathlib import Path
import sys

import torch

# Ensure backend/ is on sys.path so `import app` works when running tests via uv.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.flashvsr_service import FlashVSRService
from app.config import settings


def test_largest_8n1_leq_basic():
    helper = FlashVSRService._largest_8n1_leq

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

    s_width, s_height, t_width, t_height = FlashVSRService._compute_scaled_dims(
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
        computed = FlashVSRService._estimate_video_bytes(
            total_frames, height, width, dtype
        )
        assert computed == expected


def test_should_stream_video_respects_prefetch(monkeypatch):
    service = FlashVSRService()

    # prefetch > 0 -> streaming enabled
    monkeypatch.setattr(settings, "FLASHVSR_STREAMING_PREFETCH_FRAMES", 16, raising=False)
    assert service._should_stream_video(100, 720, 1280, torch.bfloat16) is True

    # prefetch == 0 -> streaming disabled
    monkeypatch.setattr(settings, "FLASHVSR_STREAMING_PREFETCH_FRAMES", 0, raising=False)
    assert service._should_stream_video(100, 720, 1280, torch.bfloat16) is False
