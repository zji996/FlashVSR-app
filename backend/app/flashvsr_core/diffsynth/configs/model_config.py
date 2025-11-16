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
# Minimal preset model metadata for downloader and backend service
# -----------------------------

Preset_model_id: TypeAlias = Literal[
    # WanVideo / FlashVSR presets
    "FlashVSR-1.1-Tiny-Long",
    "FlashVSR-1.1-Tiny-Long-Streaming",
    "FlashVSR-1.1-Tiny",
    "FlashVSR-1.1-Full",
]

FLASHVSR_TINY_LONG_PRESET_ID: Preset_model_id = "FlashVSR-1.1-Tiny-Long"

# 公共 ModelScope 仓库 ID（后端自动下载与 downloader 共用）
FLASHVSR_TINY_LONG_REPO_ID = "kuohao/FlashVSR-v1.1"

# 该预设下的所有权重文件（repo, 远端路径, 本地目录）
FLASHVSR_TINY_LONG_FILE_SPECS = [
    (
        FLASHVSR_TINY_LONG_REPO_ID,
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "models/FlashVSR-v1.1",
    ),
    (
        FLASHVSR_TINY_LONG_REPO_ID,
        "LQ_proj_in.ckpt",
        "models/FlashVSR-v1.1",
    ),
    (
        FLASHVSR_TINY_LONG_REPO_ID,
        "TCDecoder.ckpt",
        "models/FlashVSR-v1.1",
    ),
    (
        FLASHVSR_TINY_LONG_REPO_ID,
        "Wan2.1_VAE.pth",
        "models/FlashVSR-v1.1",
    ),
    (
        FLASHVSR_TINY_LONG_REPO_ID,
        "posi_prompt.pth",
        "models/FlashVSR-v1.1",
    ),
]

# 后端 Tiny Long 推理必需的核心文件（DiT + LQ 投影 + TCDecoder）
FLASHVSR_TINY_LONG_BASE_FILES: tuple[str, ...] = (
    "diffusion_pytorch_model_streaming_dmd.safetensors",
    "LQ_proj_in.ckpt",
    "TCDecoder.ckpt",
)

# Prompt Tensor 文件名（用于 cross-attn KV 初始化）
FLASHVSR_TINY_LONG_PROMPT_FILE = "posi_prompt.pth"

# 目前仅 Wan2.1_VAE.pth 作为额外参考文件存在（Full 变体或外部工具可用）
FLASHVSR_TINY_LONG_EXTRA_FILES: tuple[str, ...] = ("Wan2.1_VAE.pth",)

# Downloader uses these dicts to resolve preset ids into concrete files.
# Currently we only wire `FlashVSR-1.1-Tiny-Long` to the public ModelScope repo;
# other ids remain empty for future extensions.
preset_models_on_huggingface: dict[Preset_model_id, object] = {}
preset_models_on_modelscope: dict[Preset_model_id, object] = {
    FLASHVSR_TINY_LONG_PRESET_ID: {
        "file_list": FLASHVSR_TINY_LONG_FILE_SPECS,
        # ModelManager will load from these paths after download_models() returns.
        "load_path": [
            "models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors",
        ],
    },
}
