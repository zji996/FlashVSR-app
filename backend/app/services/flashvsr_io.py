"""FlashVSR 视频 IO 与权重路径辅助函数."""

from __future__ import annotations

import time
from pathlib import Path
import subprocess
from typing import Callable, Iterable, Optional

import imageio
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from app.config import settings
from app.services.video_streaming import StreamingVideoTensor


def largest_8n1_leq(n: int) -> int:
    """返回最大的 8n+1 <= n."""
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def compute_scaled_dims(
    w0: int,
    h0: int,
    scale: float,
    multiple: int = 128,
) -> tuple[int, int, int, int]:
    """
    计算缩放后的尺寸，并向下对齐到 multiple 的倍数。

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


def upscale_and_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    """放大并居中裁剪一帧图像."""
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    up = img.resize((sW, sH), Image.BICUBIC)
    l = (sW - tW) // 2
    t = (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))


def pil_to_tensor(img: Image.Image, dtype: torch.dtype, device: str) -> torch.Tensor:
    """PIL 图像转 [-1,1] 归一化 tensor (C,H,W)."""
    # 使用显式拷贝保证 NumPy 数组是可写的，避免 PyTorch 关于 non-writable tensor 的警告。
    arr = np.array(img, dtype=np.uint8, copy=True)
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return t.to(dtype)


def tensor_to_video(frames: torch.Tensor) -> list[Image.Image]:
    """将模型输出的 (C,T,H,W) tensor 转成 PIL 帧列表."""
    frames_bt = rearrange(frames, "C T H W -> T H W C")
    frames_bt = (
        (frames_bt.float() + 1)
        * 127.5
    ).clip(0, 255).cpu().numpy().astype(np.uint8)
    return [Image.fromarray(frame) for frame in frames_bt]


def save_video(
    frames: Iterable[Image.Image],
    save_path: str,
    fps: int = 30,
    quality: int = 6,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    total_frames: Optional[int] = None,
    start_time: Optional[float] = None,
) -> None:
    """将帧序列写入 MP4 文件，并按需回调进度."""
    frame_list = list(frames)
    target_total = total_frames or len(frame_list)
    begin = start_time or time.time()

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(save_path, fps=fps, quality=quality)
    try:
        for idx, frame in enumerate(tqdm(frame_list, desc="保存视频"), start=1):
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


def estimate_video_bytes(
    total_frames: int,
    height: int,
    width: int,
    dtype: torch.dtype,
) -> int:
    """估算给定 dtype 下的 LQ 视频张量占用字节数."""
    element_bytes = torch.finfo(dtype).bits // 8
    return total_frames * height * width * 3 * element_bytes


def should_stream_video(
    total_frames: int,
    height: int,
    width: int,
    dtype: torch.dtype,
) -> bool:
    """
    根据环境配置判断是否启用 LQ 流式缓冲。

    当前实现：只要预读帧数 > 0 就启用流式，限制由 FLASHVSR_STREAMING_LQ_MAX_BYTES 控制。
    """
    return settings.FLASHVSR_STREAMING_PREFETCH_FRAMES > 0


def load_frame_tensor(
    reader,
    frame_idx: int,
    scale: float,
    target_width: int,
    target_height: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """从 imageio 读取单帧并转换为模型输入 tensor."""
    img = Image.fromarray(reader.get_data(frame_idx)).convert("RGB")
    img_out = upscale_and_crop(img, scale, target_width, target_height)
    return pil_to_tensor(img_out, dtype, device)


def frame_array_to_tensor(
    frame_array,
    scale: float,
    target_width: int,
    target_height: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """将 ndarray 帧转换为模型输入 tensor."""
    img = Image.fromarray(frame_array).convert("RGB")
    img_out = upscale_and_crop(img, scale, target_width, target_height)
    return pil_to_tensor(img_out, dtype, device)


def build_streaming_video_tensor(
    reader,
    indices: list[int],
    scale: float,
    target_width: int,
    target_height: int,
    dtype: torch.dtype,
    target_device: str,
) -> StreamingVideoTensor:
    """构建 StreamingVideoTensor 以支持 LQ 流式缓冲."""
    total_needed = len(indices)
    if total_needed == 0:
        raise RuntimeError("视频没有可处理的帧")
    limit_bytes = settings.FLASHVSR_STREAMING_LQ_MAX_BYTES
    per_frame_bytes = estimate_video_bytes(1, target_height, target_width, dtype)
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
        return frame_array_to_tensor(
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


def _count_frames(reader, meta: dict) -> int:
    """计算视频总帧数（先用元数据兜底，再回退到逐帧读取）."""
    try:
        nf = meta.get("nframes", None)
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


def prepare_input(
    path: str,
    scale: float,
    device: str,
) -> tuple[torch.Tensor, int, int, int, int]:
    """读取输入视频并转换为 FlashVSR Tiny Long 所需的 LQ tensor."""
    dtype = torch.bfloat16

    reader = imageio.get_reader(path)
    first_frame = Image.fromarray(reader.get_data(0)).convert("RGB")
    w0, h0 = first_frame.size

    meta: dict = {}
    try:
        meta = reader.get_meta_data()
    except Exception:
        meta = {}

    fps_val = meta.get("fps", 30)
    fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

    total_frames = _count_frames(reader, meta)

    print(f"原始分辨率: {w0}x{h0}, 原始帧数: {total_frames}, FPS: {fps}")

    _, _, tW, tH = compute_scaled_dims(w0, h0, scale)
    print(f"目标分辨率: {tW}x{tH} (缩放 {scale}x)")

    frames: list[torch.Tensor] = []
    target_frame_device = "cpu" if device == "cuda" else device
    indices = list(range(total_frames)) + [total_frames - 1] * 4
    F = largest_8n1_leq(len(indices))
    indices = indices[:F]

    print(f"处理帧数: {F}")

    use_streaming = should_stream_video(F, tH, tW, dtype)

    reader_owned = True
    try:
        if use_streaming:
            video_tensor = build_streaming_video_tensor(
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
                    load_frame_tensor(
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


def mux_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """将已有音频流复用到指定视频文件."""
    tmp_out = str(Path(output_path).with_suffix(".muxing.tmp.mp4"))
    cmd = [
        settings.FFMPEG_BINARY,
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        tmp_out,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        cmd = [
            settings.FFMPEG_BINARY,
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            tmp_out,
        ]
        result2 = subprocess.run(cmd, capture_output=True, text=True)
        if result2.returncode != 0:
            raise RuntimeError(
                f"FFmpeg 音频合并失败: {result2.stderr.strip() or result2.stdout.strip()}"
            )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tmp_out).replace(output_path)


def merge_video_chunks(
    chunk_paths: list[Path],
    output_path: str,
    audio_path: Optional[str] = None,
) -> None:
    """将分片 MP4 文件按顺序合并为单一视频，并按需复用音频."""
    import shutil
    import subprocess

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
        # 使用上面的 mux_audio 封装
        mux_audio(str(merged_video), audio_path, output_path)
        if merged_video != Path(output_path):
            Path(merged_video).unlink(missing_ok=True)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(merged_video), output_path)
    try:
        chunk_dir.rmdir()
    except OSError:
        pass


def cleanup_chunk_artifacts(paths: list[Path]) -> None:
    """尽力删除异常情况下遗留的分片文件."""
    for path in paths:
        path.unlink(missing_ok=True)
    if paths:
        try:
            paths[0].parent.rmdir()
        except OSError:
            pass


def export_video_from_tensor(
    output_video: torch.Tensor,
    output_path: str,
    fps: int,
    total_frames: int,
    start_time: Optional[float],
    progress_callback: Optional[Callable[[int, int, float], None]],
    audio_path: Optional[str],
) -> int:
    """将内存中的输出 tensor 转换为最终 MP4 文件."""
    import shutil

    try:
        frames = tensor_to_video(output_video)
        tmp_video_only = str(Path(output_path).with_suffix(".video_only.mp4"))
        save_video(
            frames,
            tmp_video_only,
            fps=fps,
            quality=settings.FLASHVSR_EXPORT_VIDEO_QUALITY,
            progress_callback=progress_callback,
            total_frames=total_frames,
            start_time=start_time,
        )
        if audio_path and Path(audio_path).exists():
            mux_audio(tmp_video_only, audio_path, output_path)
            Path(tmp_video_only).unlink(missing_ok=True)
        else:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(tmp_video_only, output_path)
        return len(frames)
    finally:
        del output_video
