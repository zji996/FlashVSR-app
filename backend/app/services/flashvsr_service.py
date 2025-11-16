"""FlashVSR 推理服务封装."""

from __future__ import annotations

import inspect
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional

# 避免 PyTorch 预留的大块显存无法复用，默认启用可扩展分段分配。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from app.config import settings
from app.flashvsr_core.diffsynth.configs.model_config import (
    FLASHVSR_TINY_LONG_BASE_FILES,
    FLASHVSR_TINY_LONG_EXTRA_FILES,
    FLASHVSR_TINY_LONG_PROMPT_FILE,
    FLASHVSR_TINY_LONG_REPO_ID,
)
from app.services.chunk_export import ChunkedExportSession, build_chunk_base_name
from app.services.flashvsr_device import (
    resolve_device,
    decide_cache_offload_device,
    parse_pipeline_parallel,
)
from app.services.flashvsr_io import (
    prepare_input,
    export_video_from_tensor,
    merge_video_chunks,
)

# Block-Sparse 注意力依赖的 CUDA 扩展路径（实际导入在 FlashVSR pipeline 初始化时触发）
BLOCK_SPARSE_PATH = settings.THIRD_PARTY_BLOCK_SPARSE_PATH
if str(BLOCK_SPARSE_PATH) not in sys.path:
    sys.path.insert(0, str(BLOCK_SPARSE_PATH))


@dataclass
class PipelineHandle:
    """缓存的 Pipeline 实例信息."""

    variant: str
    pipeline: Any
    device: str
    default_kwargs: dict[str, Any]


class FlashVSRService:
    """FlashVSR 推理服务（单例 + 变体缓存)."""

    SUPPORTED_VARIANTS: tuple[str, ...] = ("tiny_long",)
    BASE_MODEL_FILES: tuple[str, ...] = FLASHVSR_TINY_LONG_BASE_FILES
    FULL_ONLY_FILES: tuple[str, ...] = FLASHVSR_TINY_LONG_EXTRA_FILES
    PROMPT_TENSOR_FILE = settings.FLASHVSR_PROMPT_TENSOR_PATH

    _instance: Optional["FlashVSRService"] = None
    _pipelines: dict[str, PipelineHandle] = {}
    _lock: Lock = Lock()
    _auto_download_used: bool = False
    _auto_download_source: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Pipeline 延迟加载，首次调用指定变体时再初始化
        pass

    @classmethod
    def inspect_assets(cls) -> dict[str, Any]:
        """检查模型权重情况，供系统状态和诊断使用."""

        model_path = settings.FLASHVSR_MODEL_PATH
        file_status: dict[str, bool] = {}

        for filename in cls.BASE_MODEL_FILES + cls.FULL_ONLY_FILES:
            file_status[filename] = (model_path / filename).exists()
        file_status[FLASHVSR_TINY_LONG_PROMPT_FILE] = cls.PROMPT_TENSOR_FILE.exists()

        def _ready(extra: tuple[str, ...] = ()) -> bool:
            base_ready = file_status[FLASHVSR_TINY_LONG_PROMPT_FILE] and all(
                file_status[name] for name in cls.BASE_MODEL_FILES
            )
            extra_ready = all(file_status[name] for name in extra)
            return base_ready and extra_ready

        ready_variants = {
            "tiny_long": _ready(),
        }
        missing_files = [name for name, ok in file_status.items() if not ok]

        if cls._auto_download_used:
            model_source = cls._auto_download_source or "ModelScope"
            auto_download_used = True
        else:
            auto_download_used = False
            if model_path.exists() and not missing_files:
                model_source = "local"
            else:
                model_source = None

        return {
            "model_path": str(model_path),
            "exists": model_path.exists(),
            "files": file_status,
            "ready_variants": ready_variants,
            "missing_files": missing_files,
            "auto_download_used": auto_download_used,
            "model_source": model_source,
        }

    def process_video(
        self,
        input_path: str,
        output_path: str,
        scale: float = 4.0,
        sparse_ratio: float = 2.0,
        local_range: int = 11,
        seed: int = 0,
        model_variant: str = settings.DEFAULT_MODEL_VARIANT,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        audio_path: Optional[str] = None,
    ) -> dict:
        """处理视频超分辨率."""

        variant = self._normalize_variant(model_variant)
        handle = self._get_pipeline_handle(variant)
        pipeline = handle.pipeline
        device = handle.device

        print(
            f"📹 开始处理视频: {input_path} | 模型: FlashVSR {settings.FLASHVSR_VERSION} ({variant})"
        )
        start_time = time.time()

        # 准备输入
        video_tensor, height, width, total_frames, fps = self._prepare_input(
            input_path, scale, device
        )

        print(f"📊 视频信息: {width}x{height}, {total_frames}帧, {fps}fps")

        if progress_callback and total_frames:
            progress_callback(0, total_frames, 0.0)

        # 处理视频
        infer_start = time.time()
        pipeline_kwargs = {
            "prompt": "",
            "negative_prompt": "",
            "cfg_scale": 1.0,
            "num_inference_steps": 1,
            "seed": seed,
            "LQ_video": video_tensor,
            "num_frames": total_frames,
            "height": height,
            "width": width,
            "is_full_block": False,
            "if_buffer": True,
            "topk_ratio": sparse_ratio * 768 * 1280 / (height * width),
            "kv_ratio": 3.0,
            "local_range": local_range,
            "color_fix": True,
        }
        pipeline_kwargs.update(handle.default_kwargs)

        chunk_session: Optional[ChunkedExportSession] = None
        supports_chunk_stream = "frame_chunk_handler" in inspect.signature(pipeline.__call__).parameters
        if self._should_use_chunk_writer(total_frames) and supports_chunk_stream:
            chunk_session = ChunkedExportSession(
                output_path=output_path,
                fps=fps,
                total_frames=total_frames,
                start_time=start_time,
                progress_callback=progress_callback,
                audio_path=audio_path,
            )
            pipeline_kwargs["frame_chunk_handler"] = chunk_session.handle_chunk

        cleanup_handle = video_tensor if hasattr(video_tensor, "cleanup") else None
        try:
            with torch.inference_mode():
                output_video = pipeline(**pipeline_kwargs)
        except Exception as exc:
            # 出错时不再自动导出部分结果，由上层根据 chunks_* 目录显式触发导出。
            if chunk_session:
                chunk_session.abort()
            raise
        finally:
            if cleanup_handle is not None:
                cleanup_handle.cleanup()
        inference_time = time.time() - infer_start

        if chunk_session:
            chunk_session.close()
            processed_frame_count = total_frames
        else:
            processed_frame_count = export_video_from_tensor(
                output_video=output_video,
                output_path=output_path,
                fps=fps,
                total_frames=total_frames,
                start_time=start_time,
                progress_callback=progress_callback,
                audio_path=audio_path,
            )

        total_time = time.time() - start_time

        print(f"✅ 视频处理完成: {output_path}")
        print(f"⏱️  总耗时: {total_time:.2f}秒")

        if device == "cuda":
            torch.cuda.empty_cache()

        return {
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "fps": fps,
            "processed_frames": processed_frame_count,
            "inference_time": inference_time,
            "processing_time": total_time,
            "model_variant": variant,
        }

    def _prepare_input(self, path: str, scale: float, device: str):
        """准备输入视频 tensor."""
        return prepare_input(path, scale, device)

    @staticmethod
    def _should_use_chunk_writer(total_frames: int) -> bool:
        min_frames = settings.FLASHVSR_CHUNKED_SAVE_MIN_FRAMES
        return min_frames > 0 and total_frames >= min_frames

    def _merge_video_chunks(self, chunk_paths: list[Path], output_path: str, audio_path: Optional[str] = None) -> None:
        if not chunk_paths:
            raise RuntimeError("未生成可用于合并的分片")
        chunk_paths.sort(key=lambda path: path.name)
        chunk_dir = chunk_paths[0].parent
        if len(chunk_paths) == 1:
            merged_video = chunk_paths[0]
        else:
            list_file = chunk_dir / f"{Path(output_path).stem}_chunks.txt"
            with open(list_file, "w", encoding="utf-8") as handle:
                for path in chunk_paths:
                    handle.write(f"file '{path}'\n")

            tmp_merged = chunk_dir / f"{Path(output_path).stem}_video_only.mp4"
            cmd = [
                settings.FFMPEG_BINARY,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                str(tmp_merged),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg 合并分片失败（{result.returncode}）: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )

            list_file.unlink(missing_ok=True)
            for path in chunk_paths:
                path.unlink(missing_ok=True)
            merged_video = tmp_merged

        if audio_path and Path(audio_path).exists():
            self._mux_audio(str(merged_video), audio_path, output_path)
            if merged_video != Path(output_path):
                Path(merged_video).unlink(missing_ok=True)
        else:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(merged_video), output_path)
        try:
            chunk_dir.rmdir()
        except OSError:
            pass

    @staticmethod
    def _cleanup_chunk_artifacts(paths: list[Path]) -> None:
        """尽力删除异常情况下遗留的分片文件."""
        for path in paths:
            path.unlink(missing_ok=True)
        if paths:
            try:
                paths[0].parent.rmdir()
            except OSError:
                pass

    def export_partial_from_chunks(self, expected_output_path: str) -> Optional[Path]:
        """
        基于磁盘上已有的分片文件合并并导出一个部分结果。

        - 主要用于任务已经结束（超时 / 崩溃）后，根据 chunks_* 目录中现有的分片恢复进度。
        - 不依赖仍在内存中的 ChunkedExportSession。
        """
        base_name = build_chunk_base_name(expected_output_path)
        root = settings.FLASHVSR_CHUNKED_SAVE_TMP_DIR
        if not root.exists():
            return None

        best_dir: Optional[Path] = None
        best_chunks: list[Path] = []

        for sub in root.iterdir():
            if not sub.is_dir() or not sub.name.startswith("chunks_"):
                continue
            candidates = sorted(sub.glob(f"{base_name}_chunk_*.mp4"))
            if candidates and len(candidates) > len(best_chunks):
                best_dir = sub
                best_chunks = candidates

        if not best_dir or not best_chunks:
            return None

        # 为避免最后一个未正常关闭的分片导致合并失败，保守地丢弃最后一个。
        usable = best_chunks[:-1] if len(best_chunks) > 1 else best_chunks
        if not usable:
            return None

        partial_path = Path(expected_output_path).with_name(
            f"{Path(expected_output_path).stem}_partial{Path(expected_output_path).suffix}"
        )
        # 使用与正常流程相同的合并逻辑，并在完成后清理这些分片。
        merge_video_chunks(usable, str(partial_path), audio_path=None)
        return partial_path

    def preload_variant(self, variant: Optional[str] = None) -> PipelineHandle:
        """显式预加载指定变体."""

        normalized = self._normalize_variant(variant)
        return self._get_pipeline_handle(normalized)

    def _get_pipeline_handle(self, variant: str) -> PipelineHandle:
        """获取或初始化指定变体的 pipeline."""

        if variant not in self._pipelines:
            with self._lock:
                if variant not in self._pipelines:
                    self._pipelines[variant] = self._build_pipeline_handle(variant)
        return self._pipelines[variant]

    def _build_pipeline_handle(self, variant: str) -> PipelineHandle:
        """根据变体初始化 pipeline 并缓存.

        当前实现仅支持 tiny_long 变体。
        """

        # 延迟导入 FlashVSR 相关依赖，避免在仅使用辅助方法或系统状态查询时就触发重型模型加载。
        from app.flashvsr_core import FlashVSRTinyLongPipeline, ModelManager
        from app.flashvsr_core.diffsynth.models.downloader import (
            download_customized_models,
        )
        from app.flashvsr_core.wan_utils import build_tcdecoder, Causal_LQ4x_Proj

        print(f"🚀 初始化 FlashVSR {settings.FLASHVSR_VERSION} pipeline ({variant})...")
        model_path = settings.FLASHVSR_MODEL_PATH

        needed_files = list(self.BASE_MODEL_FILES)

        missing = [name for name in needed_files if not (model_path / name).exists()]
        prompt_missing = not self.PROMPT_TENSOR_FILE.exists()

        auto_download_used = False
        model_source = "local"

        if missing or prompt_missing:
            missing_desc = ", ".join(
                missing
                + ([FLASHVSR_TINY_LONG_PROMPT_FILE] if prompt_missing else [])
            )
            print(
                f"⚠️ 检测到缺少 FlashVSR 权重文件: {missing_desc} (根目录: {model_path})，"
                f"尝试从 ModelScope 仓库 `{FLASHVSR_TINY_LONG_REPO_ID}` 自动下载..."
            )
            try:
                # 仅下载缺失部分，避免重复拉取已存在的文件。
                for filename in missing:
                    download_customized_models(
                        FLASHVSR_TINY_LONG_REPO_ID,
                        filename,
                        str(model_path),
                        downloading_priority=["ModelScope", "HuggingFace"],
                    )
                if prompt_missing:
                    download_customized_models(
                        FLASHVSR_TINY_LONG_REPO_ID,
                        FLASHVSR_TINY_LONG_PROMPT_FILE,
                        str(model_path),
                        downloading_priority=["ModelScope", "HuggingFace"],
                    )
            except Exception as exc:  # pragma: no cover - 网络/依赖错误路径
                raise FileNotFoundError(
                    "缺少 FlashVSR 权重文件，且从 ModelScope 自动下载失败，请检查网络或手动放置权重到 "
                    f"{model_path}。原始错误: {exc}"
                ) from exc

            # 自动下载后重新检查一次，确保所有必需文件均已就绪。
            missing = [name for name in needed_files if not (model_path / name).exists()]
            prompt_missing = not self.PROMPT_TENSOR_FILE.exists()
            if missing or prompt_missing:
                missing_desc = ", ".join(
                    missing
                    + ([FLASHVSR_TINY_LONG_PROMPT_FILE] if prompt_missing else [])
                )
                raise FileNotFoundError(
                    "从 ModelScope 自动下载后仍缺少 FlashVSR 权重文件: "
                    + missing_desc
                    + f" (根目录: {model_path})，请手动下载或检查路径配置。"
                )

            auto_download_used = True
            model_source = "ModelScope"

        type(self)._auto_download_used = auto_download_used
        type(self)._auto_download_source = model_source

        # 到这里权重文件已存在，本地加载模型即可。
        mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        weights_to_load = [str(model_path / self.BASE_MODEL_FILES[0])]
        mm.load_models(weights_to_load)

        if not self.PROMPT_TENSOR_FILE.exists():
            raise FileNotFoundError(
                "缺少 FlashVSR prompt tensor: "
                f"{self.PROMPT_TENSOR_FILE}. 请将 posi_prompt.pth "
                "放置在 models/FlashVSR-v1.1/ 下或通过 FLASHVSR_PROMPT_TENSOR_PATH 覆盖，详见 docs/deployment.md。"
            )

        prompt_tensor = torch.load(self.PROMPT_TENSOR_FILE, map_location="cpu")

        pipeline_cls = FlashVSRTinyLongPipeline

        device = resolve_device()
        print(f"📍 使用设备: {device}")
        if device.startswith("cuda"):
            gpu_index = 0
            try:
                if ":" in device:
                    gpu_index = int(device.split(":", 1)[1])
            except Exception:
                gpu_index = 0
            try:
                print(f"🎮 GPU: {torch.cuda.get_device_name(gpu_index)}")
            except Exception:
                pass

        pipe = pipeline_cls.from_model_manager(mm, device=device)

        cache_device, cache_reason = decide_cache_offload_device(device)
        pipe.set_cache_offload_device(cache_device)
        if cache_device:
            print(f"💾 KV cache offload → {cache_device} ({cache_reason})")

        # 配置 LQ 投影层
        lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(
            device, dtype=torch.bfloat16
        )
        lq_proj.load_state_dict(
            torch.load(model_path / "LQ_proj_in.ckpt", map_location="cpu"),
            strict=True,
        )
        lq_proj.to(device)
        pipe.denoising_model().LQ_proj_in = lq_proj

        # 配置 TCDecoder
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(
            new_channels=multi_scale_channels, new_latent_channels=16 + 768
        )
        pipe.TCDecoder.load_state_dict(
            torch.load(model_path / "TCDecoder.ckpt", map_location="cpu"),
            strict=False,
        )

        # 可选：流水线并行（多 GPU）
        pp_devices, pp_split = parse_pipeline_parallel()

        default_kwargs: dict[str, Any] = {}

        pipe.to(device)
        # 启用流水线并行时，不启用 VRAM management 避免设备错配
        if pp_devices is None:
            pipe.enable_vram_management(num_persistent_param_in_dit=None)
        pipe.init_cross_kv(context_tensor=prompt_tensor)
        pipe.load_models_to_device(["dit", "vae"])

        # 初始化流水线并行（需要在 init_cross_kv 之后，把 cross-attn 缓存也迁移）
        if pp_devices is not None and hasattr(pipe, "enable_pipeline_parallel"):
            try:
                pipe.enable_pipeline_parallel(pp_devices, split_index=pp_split)
                print(f"🔀 Pipeline parallel enabled on {pp_devices} (split @ block {pp_split if pp_split is not None else 'auto'})")
                # 传递重叠调度模式给 pipeline（若实现）
                try:
                    overlap_mode = getattr(settings, "FLASHVSR_PP_OVERLAP_MODE", "basic")
                    if hasattr(pipe, "pp_overlap_mode"):
                        pipe.pp_overlap_mode = (overlap_mode or "basic").lower()
                        print(f"⚙️ Pipeline overlap mode set to {pipe.pp_overlap_mode}")
                except Exception:
                    pass
            except Exception as e:
                print(f"⚠️ 启用流水线并行失败：{e}")
            # When PP is enabled, move TCDecoder to the last stage device to free GPU0 for Stage0
            try:
                dev1 = pp_devices[-1]
                if hasattr(pipe, "TCDecoder") and pipe.TCDecoder is not None:
                    pipe.TCDecoder.to(dev1)
                    print(f"🎯 TCDecoder moved to {dev1} for overlap")
            except Exception as e:
                print(f"⚠️ TCDecoder 迁移到 {pp_devices[-1]} 失败：{e}")
        # Overlap scheduling for single video (optional)
        try:
            if getattr(settings, "FLASHVSR_PP_OVERLAP", False) and hasattr(pipe, "enable_pipeline_overlap"):
                pipe.enable_pipeline_overlap(True)
                print("⏩ Pipeline overlap (Stage0/Stage1) enabled per window")
        except Exception as e:
            print(f"⚠️ 启用流水线重叠失败：{e}")

        print(f"✅ FlashVSR pipeline ({variant}) 初始化完成")
        return PipelineHandle(
            variant=variant,
            pipeline=pipe,
            device=device,
            default_kwargs=default_kwargs,
        )

    def _normalize_variant(self, variant: Optional[str]) -> str:
        value = (variant or settings.DEFAULT_MODEL_VARIANT).lower()
        if value not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"不支持的模型变体: {variant}. 可选: {', '.join(self.SUPPORTED_VARIANTS)}"
            )
        asset_status = self.inspect_assets().get("ready_variants", {})
        if not asset_status.get(value, False):
            raise RuntimeError(
                f"模型变体 {value} 缺少必要权重，请参考 README 下载 FlashVSR {settings.FLASHVSR_VERSION} 权重"
            )
        return value
