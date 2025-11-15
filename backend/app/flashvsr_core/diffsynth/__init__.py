"""
Minimal diffsynth surface for backend FlashVSR usage.

We only re-export the pieces needed by the service layer:
- `models` (for `ModelManager` and WanVideo modules)
- `pipelines` (for `FlashVSRTinyLongPipeline`)
"""

from .models import *  # noqa: F401,F403
from .pipelines import *  # noqa: F401,F403
