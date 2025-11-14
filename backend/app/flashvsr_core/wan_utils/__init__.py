"""WanVSR helper modules (TCDecoder, LQ projection).

This vendors the minimal pieces from `examples/WanVSR/utils` needed by
the backend so that we do not import from `third_party/FlashVSR`.
"""

from .tcdecoder import build_tcdecoder  # noqa: F401
from .lq_proj import Causal_LQ4x_Proj  # noqa: F401

