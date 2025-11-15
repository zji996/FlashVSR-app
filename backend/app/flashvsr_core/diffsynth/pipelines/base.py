import torch
from typing import Optional


class BasePipeline(torch.nn.Module):
    """Minimal base pipeline with device helpers and streaming-aware LQ access."""

    def __init__(
        self,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        height_division_factor: int = 64,
        width_division_factor: int = 64,
    ) -> None:
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.cpu_offload = False
        self.model_names: list[str] = []
        self.cache_offload_device: Optional[str] = None
        # Pipeline parallel config (optional)
        self.pp_devices: Optional[list[str]] = None
        self.pp_split_idx: Optional[int] = None
        self.pp_overlap: bool = False

    def check_resize_height_width(self, height: int, width: int) -> tuple[int, int]:
        """Round height/width up to divisors required by WanVideo windows."""
        if height % self.height_division_factor != 0:
            height = (
                (height + self.height_division_factor - 1)
                // self.height_division_factor
                * self.height_division_factor
            )
            print(
                f"The height cannot be evenly divided by {self.height_division_factor}. "
                f"We round it up to {height}."
            )
        if width % self.width_division_factor != 0:
            width = (
                (width + self.width_division_factor - 1)
                // self.width_division_factor
                * self.width_division_factor
            )
            print(
                f"The width cannot be evenly divided by {self.width_division_factor}. "
                f"We round it up to {width}."
            )
        return height, width

    def enable_cpu_offload(self) -> None:
        self.cpu_offload = True

    def set_cache_offload_device(self, device: Optional[str] = "cpu"):
        """
        Control where streamed KV caches live between steps.
        None keeps them on the compute device, while 'cpu' spills them to host RAM.
        """
        self.cache_offload_device = device

    # ---- Pipeline parallel (no-op default) ----
    def enable_pipeline_parallel(self, devices: list[str], split_index: Optional[int] = None):
        """Enable pipeline parallelism. Derived pipelines may override to place blocks across devices."""
        self.pp_devices = list(devices)
        self.pp_split_idx = split_index
        return self

    def enable_pipeline_overlap(self, enabled: bool = True):
        """Enable overlapped scheduling across windows when pipeline parallel is on."""
        self.pp_overlap = bool(enabled)
        return self

    def _fetch_lq_clip(
        self,
        lq_video,
        start: int,
        end: int,
        *,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
    ):
        """
        Retrieve an LQ frame slice either from an in-memory tensor or a lazy backend.
        """
        if lq_video is None:
            return None

        device = device or self.device
        dtype = dtype or self.torch_dtype

        if hasattr(lq_video, "get_clip"):
            return lq_video.get_clip(
                start,
                end,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )

        start = max(start, 0)
        end = min(end, lq_video.shape[2])
        if end <= start:
            return lq_video[:, :, start:start, :, :].to(device=device, dtype=dtype)
        clip = lq_video[:, :, start:end, :, :]
        return clip.to(device=device, dtype=dtype, non_blocking=non_blocking)

    def _release_lq_frames(self, lq_video, upto: int) -> None:
        """
        Hint the LQ provider that frames prior to `upto` are no longer needed.
        Streaming backends can use this to free host RAM.
        """
        if lq_video is None:
            return
        release_cb = getattr(lq_video, "release_until", None)
        if release_cb is None:
            return
        try:
            release_cb(upto)
        except Exception:
            pass


    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    model.to(self.device)
        # fresh the cuda cache
        torch.cuda.empty_cache()

    
    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise
