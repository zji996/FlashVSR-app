"""Local FlashVSR core bindings.

This package vendors the upstream `diffsynth` stack needed for FlashVSR
so that the backend does not depend on `third_party/FlashVSR` at runtime.
"""

from .diffsynth.models import ModelManager  # noqa: F401
from .diffsynth.pipelines import (  # noqa: F401
    FlashVSRTinyLongPipeline,
)

