from typing_extensions import Literal, TypeAlias

from ..models.wan_video_dit import WanModel
from ..models.wan_video_vae import WanVideoVAE


# -----------------------------
# Minimal model loader configs for WanVideo / FlashVSR Tiny Long
# -----------------------------

# These configs are used by ModelManager to detect model type automatically.
# Format: (state_dict_keys_hash, state_dict_keys_with_shape, model_names, model_classes, model_resource)
model_loader_configs = [
    # WanVideo DiT (base video diffusion)
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
    # Diffusers-format WanVideo DiT
    (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),
    # WanVideo VAE (Tiny Long decoder)
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
]

# For the backend service we do not auto-load HuggingFace or patch models,
# but ModelManager expects these symbols to exist.
huggingface_model_loader_configs: list = []
patch_model_loader_configs: list = []


# -----------------------------
# Minimal preset model metadata for downloader (unused in backend service)
# -----------------------------

Preset_model_id: TypeAlias = Literal[
    # WanVideo / FlashVSR presets (placeholder; backend does not call download_models)
    "FlashVSR-1.1-Tiny-Long",
    "FlashVSR-1.1-Tiny-Long-Streaming",
    "FlashVSR-1.1-Tiny",
    "FlashVSR-1.1-Full",
]

# Downloader expects these dicts to exist; keep them empty to avoid unused downloads.
preset_models_on_huggingface: dict[Preset_model_id, object] = {}
preset_models_on_modelscope: dict[Preset_model_id, object] = {}
