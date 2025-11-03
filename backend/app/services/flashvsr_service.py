"""FlashVSR 推理服务封装."""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Callable

import torch
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
from einops import rearrange

from app.config import settings

# 添加FlashVSR到Python路径
THIRD_PARTY_ROOT = Path("/app/third_party")
FLASHVSR_PATH = THIRD_PARTY_ROOT / "FlashVSR"
if str(FLASHVSR_PATH) not in sys.path:
    sys.path.insert(0, str(FLASHVSR_PATH))

from diffsynth import ModelManager, FlashVSRTinyPipeline

# 导入FlashVSR工具函数
WANVSR_PATH = FLASHVSR_PATH / "examples" / "WanVSR"
if str(WANVSR_PATH) not in sys.path:
    sys.path.insert(0, str(WANVSR_PATH))

from utils.utils import Buffer_LQ4x_Proj
from utils.TCDecoder import build_tcdecoder


class FlashVSRService:
    """FlashVSR推理服务（单例模式）."""
    
    _instance: Optional['FlashVSRService'] = None
    _pipeline = None
    
    def __new__(cls):
        """单例模式."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化."""
        if self._pipeline is None:
            self._init_pipeline()
    
    def _init_pipeline(self):
        """初始化FlashVSR pipeline."""
        print("🚀 正在初始化 FlashVSR pipeline...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📍 使用设备: {device}")
        
        if device == "cuda":
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        
        # 加载模型
        mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_path = settings.FLASHVSR_MODEL_PATH

        mm.load_models(
            [
                str(model_path / "diffusion_pytorch_model_streaming_dmd.safetensors"),
            ]
        )
        
        self._pipeline = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        
        # 加载LQ投影层
        self._pipeline.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(
            in_dim=3, out_dim=1536, layer_num=1
        ).to(device, dtype=torch.bfloat16)
        
        lq_proj_path = model_path / "LQ_proj_in.ckpt"
        if lq_proj_path.exists():
            self._pipeline.denoising_model().LQ_proj_in.load_state_dict(
                torch.load(lq_proj_path, map_location="cpu"), strict=True
            )
        self._pipeline.denoising_model().LQ_proj_in.to(device)
        
        # 加载TCDecoder
        multi_scale_channels = [512, 256, 128, 128]
        self._pipeline.TCDecoder = build_tcdecoder(
            new_channels=multi_scale_channels, new_latent_channels=16 + 768
        )
        self._pipeline.TCDecoder.load_state_dict(
            torch.load(model_path / "TCDecoder.ckpt"), strict=False
        )
        
        self._pipeline.to(device)
        self._pipeline.enable_vram_management(num_persistent_param_in_dit=None)
        self._pipeline.init_cross_kv()
        self._pipeline.load_models_to_device(["dit", "vae"])
        
        print("✅ FlashVSR pipeline 初始化完成")
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        scale: float = 4.0,
        sparse_ratio: float = 2.0,
        local_range: int = 11,
        seed: int = 0,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> dict:
        """
        处理视频超分辨率.
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            scale: 超分倍数
            sparse_ratio: 稀疏比率
            local_range: 局部范围
            seed: 随机种子
            progress_callback: 进度回调函数(processed_frames, total_frames, avg_time)
        
        Returns:
            包含视频信息的字典
        """
        print(f"📹 开始处理视频: {input_path}")
        start_time = time.time()
        
        # 准备输入
        video_tensor, height, width, total_frames, fps = self._prepare_input(
            input_path, scale
        )
        
        print(f"📊 视频信息: {width}x{height}, {total_frames}帧, {fps}fps")
        
        # 超分处理
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if progress_callback and total_frames:
            progress_callback(0, total_frames, 0.0)

        # 处理视频
        infer_start = time.time()
        output_video = self._pipeline(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=seed,
            LQ_video=video_tensor,
            num_frames=total_frames,
            height=height,
            width=width,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=sparse_ratio * 768 * 1280 / (height * width),
            kv_ratio=3.0,
            local_range=local_range,
            color_fix=True,
        )
        inference_time = time.time() - infer_start
        
        # 转换为视频帧
        frames = self._tensor2video(output_video)
        
        # 保存视频
        self._save_video(
            frames,
            output_path,
            fps=fps,
            progress_callback=progress_callback,
            total_frames=total_frames,
            start_time=start_time,
        )
        
        total_time = time.time() - start_time
        
        print(f"✅ 视频处理完成: {output_path}")
        print(f"⏱️  总耗时: {total_time:.2f}秒")
        
        # 清理GPU缓存
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return {
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "fps": fps,
            "processed_frames": len(frames),
            "inference_time": inference_time,
            "processing_time": total_time,
        }
    
    def _prepare_input(self, path: str, scale: float):
        """准备输入视频tensor."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16
        
        # 读取视频
        reader = imageio.get_reader(path)
        first_frame = Image.fromarray(reader.get_data(0)).convert('RGB')
        w0, h0 = first_frame.size
        
        # 获取元数据
        meta = {}
        try:
            meta = reader.get_meta_data()
        except:
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
        indices = list(range(total_frames)) + [total_frames - 1] * 4
        F = self._largest_8n1_leq(len(indices))
        indices = indices[:F]
        
        print(f"处理帧数: {F}")
        
        try:
            for i in tqdm(indices, desc="加载视频帧"):
                img = Image.fromarray(reader.get_data(i)).convert('RGB')
                img_out = self._upscale_and_crop(img, scale, tW, tH)
                frames.append(self._pil_to_tensor(img_out, dtype, device))
        finally:
            reader.close()
        
        video_tensor = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
        return video_tensor, tH, tW, F, fps
    
    @staticmethod
    def _count_frames(reader, meta):
        """计算视频总帧数."""
        try:
            nf = meta.get('nframes', None)
            if isinstance(nf, int) and nf > 0:
                return nf
        except:
            pass
        
        try:
            return reader.count_frames()
        except:
            n = 0
            try:
                while True:
                    reader.get_data(n)
                    n += 1
            except:
                return n
    
    @staticmethod
    def _compute_scaled_dims(w0: int, h0: int, scale: float, multiple: int = 128):
        """计算缩放后的尺寸."""
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
        """PIL图像转tensor."""
        t = torch.from_numpy(np.asarray(img, np.uint8)).to(
            device=device, dtype=torch.float32
        )
        t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        return t.to(dtype)
    
    @staticmethod
    def _tensor2video(frames):
        """Tensor转视频帧."""
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
    
    @staticmethod
    def _largest_8n1_leq(n: int) -> int:
        """返回最大的 8n+1 <= n."""
        return 0 if n < 1 else ((n - 1) // 8) * 8 + 1
