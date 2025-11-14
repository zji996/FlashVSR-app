"""FlashVSR 推理服务封装."""

from __future__ import annotations

import inspect
import os
import sys
import time
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional

# 避免 PyTorch 预留的大块显存无法复用，默认启用可扩展分段分配。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import imageio
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from app.config import settings
from app.services.chunk_export import ChunkedExportSession
from app.services.video_streaming import StreamingVideoTensor
from app.flashvsr_core import FlashVSRTinyLongPipeline, ModelManager
from app.flashvsr_core.wan_utils import build_tcdecoder, Causal_LQ4x_Proj

# Block-Sparse 注意力依赖的 CUDA 扩展
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
    BASE_MODEL_FILES: tuple[str, ...] = (
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt",
    )
    FULL_ONLY_FILES: tuple[str, ...] = ("Wan2.1_VAE.pth",)
    PROMPT_TENSOR_FILE = settings.FLASHVSR_PROMPT_TENSOR_PATH

    _instance: Optional["FlashVSRService"] = None
    _pipelines: dict[str, PipelineHandle] = {}
    _lock: Lock = Lock()

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
        file_status["posi_prompt.pth"] = cls.PROMPT_TENSOR_FILE.exists()

        def _ready(extra: tuple[str, ...] = ()) -> bool:
            base_ready = file_status["posi_prompt.pth"] and all(
                file_status[name] for name in cls.BASE_MODEL_FILES
            )
            extra_ready = all(file_status[name] for name in extra)
            return base_ready and extra_ready

        ready_variants = {
            "tiny_long": _ready(),
        }
        missing_files = [name for name, ok in file_status.items() if not ok]

        return {
            "model_path": str(model_path),
            "exists": model_path.exists(),
            "files": file_status,
            "ready_variants": ready_variants,
            "missing_files": missing_files,
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
                service=self,
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
        except Exception:
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
            processed_frame_count = self._export_video(
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
        dtype = torch.bfloat16

        # 读取视频
        reader = imageio.get_reader(path)
        first_frame = Image.fromarray(reader.get_data(0)).convert('RGB')
        w0, h0 = first_frame.size

        # 获取元数据
        meta = {}
        try:
            meta = reader.get_meta_data()
        except Exception:
            pass

        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        # 获取总帧数
        total_frames = self._count_frames(reader, meta)

        print(f"原始分辨率: {w0}x{h0}, 原始帧数: {total_frames}, FPS: {fps}")

        # 计算目标尺寸
        sW, sH, tW, tH = self._compute_scaled_dims(w0, h0, scale)
        print(f"目标分辨率: {tW}x{tH} (缩放 {scale}x)")

        # 读取所有帧
        frames = []
        # 当推理在 GPU 上运行时，不要在这里把所有帧一次性搬到显存。
        # 先推进 CPU，后续按块 `.to(self.device)`，可以把 ~4GB 的 3K@105 帧常驻显存消除掉。
        target_frame_device = "cpu" if device == "cuda" else device
        indices = list(range(total_frames)) + [total_frames - 1] * 4
        F = self._largest_8n1_leq(len(indices))
        indices = indices[:F]

        print(f"处理帧数: {F}")

        use_streaming = self._should_stream_video(F, tH, tW, dtype)

        reader_owned = True
        try:
            if use_streaming:
                video_tensor = self._build_streaming_video_tensor(
                    reader,
                    indices,
                    scale,
                    tW,
                    tH,
                    dtype,
                    target_frame_device,
                )
                reader_owned = False
            else:
                for i in tqdm(indices, desc="加载视频帧"):
                    frames.append(
                        self._load_frame_tensor(
                            reader,
                            i,
                            scale,
                            tW,
                            tH,
                            dtype,
                            target_frame_device,
                        )
                    )
                video_tensor = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
                if device == "cuda":
                    video_tensor = video_tensor.pin_memory()
        finally:
            if reader_owned:
                reader.close()

        return video_tensor, tH, tW, F, fps

    @staticmethod
    def _count_frames(reader, meta):
        """计算视频总帧数."""
        try:
            nf = meta.get('nframes', None)
            if isinstance(nf, int) and nf > 0:
                return nf
        except Exception:
            pass

        try:
            return reader.count_frames()
        except Exception:
            n = 0
            try:
                while True:
                    reader.get_data(n)
                    n += 1
            except Exception:
                return n

    @staticmethod
    def _compute_scaled_dims(w0: int, h0: int, scale: float, multiple: int = 128):
        """
        计算缩放后的尺寸。

        - 先按 scale 计算放大后的尺寸 (sW, sH)。
        - 再向下对齐到 multiple 的倍数，用于满足 FlashVSR 的块大小约束。
        - 这里保持与官方 FlashVSR WanVideo 模型一致，使用 multiple=128；
          这样在 VAE 下采样 (×1/8) 和 3D patch (1,2,2) 之后，特征图尺寸依然能被
          self-attention 的窗口 (2,8,8) 整除，避免 “Dims must divide by window size” 错误。
        """
        sW = int(round(w0 * scale))
        sH = int(round(h0 * scale))

        tW = (sW // multiple) * multiple
        tH = (sH // multiple) * multiple

        return sW, sH, tW, tH

    @staticmethod
    def _upscale_and_crop(img: Image.Image, scale: float, tW: int, tH: int):
        """放大并居中裁剪."""
        w0, h0 = img.size
        sW = int(round(w0 * scale))
        sH = int(round(h0 * scale))

        up = img.resize((sW, sH), Image.BICUBIC)
        l = (sW - tW) // 2
        t = (sH - tH) // 2
        return up.crop((l, t, l + tW, t + tH))

    @staticmethod
    def _pil_to_tensor(img: Image.Image, dtype, device):
        """PIL 图像转 tensor."""
        # 使用显式拷贝保证 NumPy 数组是可写的，避免 PyTorch 关于
        # "non-writable tensor" 的警告，同时保持 dtype/layout 不变。
        arr = np.array(img, dtype=np.uint8, copy=True)
        t = torch.from_numpy(arr).to(
            device=device, dtype=torch.float32
        )
        t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        return t.to(dtype)

    @staticmethod
    def _tensor2video(frames):
        """Tensor 转视频帧."""
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @staticmethod
    def _save_video(
        frames,
        save_path: str,
        fps: int = 30,
        quality: int = 6,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        total_frames: Optional[int] = None,
        start_time: Optional[float] = None,
    ):
        """保存视频."""
        target_total = total_frames or len(frames)
        begin = start_time or time.time()

        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        writer = imageio.get_writer(save_path, fps=fps, quality=quality)
        try:
            for idx, frame in enumerate(tqdm(frames, desc="保存视频"), start=1):
                writer.append_data(np.array(frame))
                if progress_callback:
                    elapsed = max(time.time() - begin, 0.0)
                    avg_time = elapsed / idx if idx else 0.0
                    progress_callback(min(idx, target_total), target_total, avg_time)
        finally:
            writer.close()

        if progress_callback:
            elapsed = max(time.time() - begin, 0.0)
            avg_time = elapsed / target_total if target_total else 0.0
            progress_callback(target_total, target_total, avg_time)

    def _export_video(
        self,
        output_video,
        output_path: str,
        fps: int,
        total_frames: int,
        start_time: Optional[float],
        progress_callback: Optional[Callable[[int, int, float], None]],
        audio_path: Optional[str],
    ) -> int:
        """Convert the in-memory tensor into a video file."""
        try:
            frames = self._tensor2video(output_video)
            tmp_video_only = str(Path(output_path).with_suffix(".video_only.mp4"))
            self._save_video(
                frames,
                tmp_video_only,
                fps=fps,
                quality=settings.FLASHVSR_EXPORT_VIDEO_QUALITY,
                progress_callback=progress_callback,
                total_frames=total_frames,
                start_time=start_time,
            )
            if audio_path and Path(audio_path).exists():
                self._mux_audio(tmp_video_only, audio_path, output_path)
                Path(tmp_video_only).unlink(missing_ok=True)
            else:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(tmp_video_only, output_path)
            return len(frames)
        finally:
            del output_video

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
    @staticmethod
    def _mux_audio(video_path: str, audio_path: str, output_path: str) -> None:
        """Mux existing audio into the given video file."""
        tmp_out = str(Path(output_path).with_suffix(".muxing.tmp.mp4"))
        cmd = [
            settings.FFMPEG_BINARY,
            "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "copy",
            "-shortest",
            tmp_out,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback: transcode audio to AAC
            cmd = [
                settings.FFMPEG_BINARY,
                "-y",
                "-i", video_path,
                "-i", audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                tmp_out,
            ]
            result2 = subprocess.run(cmd, capture_output=True, text=True)
            if result2.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg 音频合并失败: {result2.stderr.strip() or result2.stdout.strip()}"
                )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(tmp_out, output_path)
    @staticmethod
    def _largest_8n1_leq(n: int) -> int:
        """返回最大的 8n+1 <= n."""
        return 0 if n < 1 else ((n - 1) // 8) * 8 + 1

    @staticmethod
    def _estimate_video_bytes(total_frames: int, height: int, width: int, dtype: torch.dtype) -> int:
        element_bytes = torch.finfo(dtype).bits // 8
        return total_frames * height * width * 3 * element_bytes

    def _should_stream_video(self, total_frames: int, height: int, width: int, dtype: torch.dtype) -> bool:
        # Streaming is always enabled as long as prefetch > 0. Use the env limit only to cap buffer size.
        return settings.FLASHVSR_STREAMING_PREFETCH_FRAMES > 0

    def _load_frame_tensor(
        self,
        reader,
        frame_idx: int,
        scale: float,
        target_width: int,
        target_height: int,
        dtype: torch.dtype,
        device: str,
    ) -> torch.Tensor:
        img = Image.fromarray(reader.get_data(frame_idx)).convert('RGB')
        img_out = self._upscale_and_crop(img, scale, target_width, target_height)
        return self._pil_to_tensor(img_out, dtype, device)

    def _frame_array_to_tensor(
        self,
        frame_array,
        scale: float,
        target_width: int,
        target_height: int,
        dtype: torch.dtype,
        device: str,
    ) -> torch.Tensor:
        img = Image.fromarray(frame_array).convert('RGB')
        img_out = self._upscale_and_crop(img, scale, target_width, target_height)
        return self._pil_to_tensor(img_out, dtype, device)

    def _build_streaming_video_tensor(
        self,
        reader,
        indices: list[int],
        scale: float,
        target_width: int,
        target_height: int,
        dtype: torch.dtype,
        target_device: str,
    ) -> StreamingVideoTensor:
        total_needed = len(indices)
        if total_needed == 0:
            raise RuntimeError("视频没有可处理的帧")
        limit_bytes = settings.FLASHVSR_STREAMING_LQ_MAX_BYTES
        per_frame_bytes = self._estimate_video_bytes(1, target_height, target_width, dtype)
        if per_frame_bytes <= 0:
            raise RuntimeError("无法计算单帧缓冲大小")
        if limit_bytes <= 0:
            frames_from_limit = total_needed
        else:
            frames_from_limit = limit_bytes // per_frame_bytes
            if frames_from_limit <= 0:
                raise RuntimeError(
                    "FLASHVSR_STREAMING_LQ_MAX_BYTES 太小，连单帧 LQ 缓冲都容纳不了"
                )
        prefetch = max(1, min(settings.FLASHVSR_STREAMING_PREFETCH_FRAMES, total_needed))
        if frames_from_limit < prefetch:
            required_bytes = prefetch * per_frame_bytes
            raise RuntimeError(
                "FLASHVSR_STREAMING_LQ_MAX_BYTES 太小，无法预读启动推理所需的帧数；"
                f"至少需要 {required_bytes / (1024**3):.2f} GB 才能缓存 {prefetch} 帧"
            )
        capacity_frames = min(frames_from_limit, total_needed)

        def _read(idx: int):
            return reader.get_data(idx)

        def _process(frame_array) -> torch.Tensor:
            return self._frame_array_to_tensor(
                frame_array,
                scale,
                target_width,
                target_height,
                dtype,
                target_device,
            )

        return StreamingVideoTensor(
            reader=reader,
            indices=list(indices),
            read_frame_fn=_read,
            process_frame_fn=_process,
            height=target_height,
            width=target_width,
            dtype=dtype,
            max_buffer_frames=capacity_frames,
            prefetch_frames=prefetch,
            per_frame_bytes=per_frame_bytes,
            target_device=target_device,
            decode_workers=settings.FLASHVSR_STREAMING_DECODE_THREADS,
            lock_memory=limit_bytes > 0,
        )

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

        print(f"🚀 初始化 FlashVSR {settings.FLASHVSR_VERSION} pipeline ({variant})...")
        model_path = settings.FLASHVSR_MODEL_PATH

        needed_files = list(self.BASE_MODEL_FILES)

        missing = [name for name in needed_files if not (model_path / name).exists()]
        if missing:
            raise FileNotFoundError(
                "缺少 FlashVSR 权重文件: " + ", ".join(missing) + f" (根目录: {model_path})"
            )

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

        device = self._resolve_device()
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

        cache_device, cache_reason = self._decide_cache_offload_device(device)
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
        pp_devices, pp_split = self._parse_pipeline_parallel()

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

    def _decide_cache_offload_device(self, device: str) -> tuple[Optional[str], Optional[str]]:
        """
        Determine whether to spill streaming KV caches to CPU, returning (device, reason).
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

        # Query the correct GPU properties if a specific index is requested
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
            return (
                "cpu",
                f"auto: GPU {total_gb:.1f} GB ≤ {threshold:.1f} GB",
            )
        return None, None

    def _resolve_device(self) -> str:
        """Resolve target torch device from settings and availability."""
        override = (settings.FLASHVSR_DEVICE or "").strip()
        if override:
            if override.startswith("cuda"):
                if torch.cuda.is_available():
                    # Optionally set current device if index provided
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

    def _parse_pipeline_parallel(self) -> tuple[Optional[list[str]], Optional[int]]:
        """Parse pipeline-parallel settings from env Settings.
        Returns (devices, split_index) or (None, None) if disabled.
        """
        raw = (settings.FLASHVSR_PP_DEVICES or "").strip()
        if not raw:
            return None, None
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        devices: list[str] = []
        for p in parts:
            if p.startswith("cuda"):
                devices.append(p)
            elif p.isdigit():
                devices.append(f"cuda:{p}")
            else:
                # fallback: accept 'cpu' or unknown
                devices.append(p)
        # Need at least 2 devices
        if len(devices) < 2:
            return None, None

        split_raw = (settings.FLASHVSR_PP_SPLIT_BLOCK or "auto").strip().lower()
        split_index: Optional[int]
        if split_raw in ("", "auto"):
            split_index = None
        else:
            try:
                split_index = int(split_raw)
            except Exception:
                split_index = None
        return devices, split_index
