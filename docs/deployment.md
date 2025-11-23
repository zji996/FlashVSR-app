# 部署指南

部署分为本地开发环境与 Docker Compose 全栈两条路径，关键依赖为 Python 3.12+、Node 20+、Git LFS（或手动下载权重），以及可选的 NVIDIA Container Toolkit。

## 本地开发流程

1. **安装依赖**
   ```bash
   # 1）在 backend 目录下手动创建并初始化虚拟环境（推荐 3.12）
   cd backend
   uv venv --python 3.12            # 创建 backend/.venv
   source .venv/bin/activate        # 激活虚拟环境
   uv sync                          # 安装后端依赖到当前 venv

   # 2）安装前端依赖
   cd ../frontend
   pnpm install
   ```

   上面的命令会：

   - 在 `backend/` 下创建一个显式的虚拟环境目录 `backend/.venv`；
   - 使用 `backend/pyproject.toml` 中声明的依赖，将后端运行/开发依赖安装到该虚拟环境中；
   - 前端依赖安装在 `frontend/node_modules`。

   本地开发时常见的虚拟环境 / uv 用法示例：

   ```bash
   # 激活 backend 虚拟环境后：
   cd backend
   source .venv/bin/activate

   fastapi dev app/main.py          # 启动后端 API（开发模式）
   celery -A app.core.celery_app worker --loglevel=info --concurrency=1
   pytest                           # 运行后端测试

   ```

   **构建 Block-Sparse-Attention CUDA 扩展（必需）**

   推荐在仓库根目录直接执行一条命令（避免相对路径出错），脚本会自动：
   - 检查 `backend/.venv` 中是否已能导入 `block_sparse_attn`，如已安装则直接跳过构建；
   - 按物理内存自动选择较保守的 `MAX_JOBS` / `NINJAFLAGS`，降低 OOM 和“卡死”概率。

   ```bash
   bash scripts/install_block_sparse_attn.sh
   ```

   如需强制重新编译（例如升级了 CUDA / PyTorch）：

   ```bash
   bash scripts/install_block_sparse_attn.sh --force
   ```

   若不想使用脚本，也可以在仓库根目录手动执行等价命令（非 editable 安装，后续可复用已构建的 wheel，加快重新安装速度）：

   ```bash
   uv pip install \
     --python backend/.venv/bin/python \
     --no-build-isolation \
     third_party/Block-Sparse-Attention
   ```

   该命令会在 `third_party/Block-Sparse-Attention` 中构建 `block_sparse_attn_cuda`，用于 WanVSR 模型的稀疏注意力推理；需要本地 CUDA 11.6+ 和 NVCC。

   若你想手动控制并发度（而不是使用脚本的自动选择），可以在执行命令前自行设置：
   - 物理内存 16 GB 主机：`export MAX_JOBS=3`、`export NINJAFLAGS=-j3`
   - 物理内存 32 GB 主机：`export MAX_JOBS=6`、`export NINJAFLAGS=-j6`
   - 物理内存 32+ GB 主机：`export MAX_JOBS=8`、`export NINJAFLAGS=-j8`
   - 如仍遇到 OOM，可进一步下调上述数值至 2 或 1。

   如果提示 `The detected CUDA version ... mismatches ... torch.version.cuda`，需安装与当前 PyTorch (`torch.version.cuda` 显示的版本) 一致的 CUDA Toolkit，或改用匹配版本的 PyTorch wheel。
2. **准备模型权重**

   默认约定 Tiny Long 权重放在 `backend/models/FlashVSR-v1.1`，所需文件为：

   - `diffusion_pytorch_model_streaming_dmd.safetensors`
   - `LQ_proj_in.ckpt`
   - `TCDecoder.ckpt`
   - `Wan2.1_VAE.pth`
   - `posi_prompt.pth`

   当前仓库内置了对 ModelScope 上公开模型的自动下载支持：

   - 模型仓库：`kuohao/FlashVSR-v1.1`  
   - 依赖：`modelscope`（已在 `backend` 的依赖中声明）

   - **全自动模式（推荐）**：当首次真正初始化 FlashVSR pipeline 时（第一次收到超分任务），后端会检测本地是否存在上述权重；如果缺失，会在日志中打印提示，并自动从 ModelScope 仓库 `kuohao/FlashVSR-v1.1` 下载到 `backend/models/FlashVSR-v1.1`。前提是部署环境能访问外网且已安装 `modelscope`。

   - **显式预下载（可选）**：如果你想在正式服务前就把权重拉好，可以手动触发一次下载（推荐在项目根目录执行）：

   ```bash
   cd backend
   uv run python - << 'PY'
   from app.flashvsr_core import ModelManager

   # 通过预设 id 从 ModelScope 下载 Tiny Long 权重到 backend/models/FlashVSR-v1.1
   mm = ModelManager(
       torch_dtype="bf16",
       device="cpu",
       model_id_list=["FlashVSR-1.1-Tiny-Long"],
   )
   print("Downloaded models:", mm.model_name)
   PY
   ```

   下载完成后，目录结构类似：

   ```bash
   ls backend/models/FlashVSR-v1.1
   # diffusion_pytorch_model_streaming_dmd.safetensors
   # LQ_proj_in.ckpt
   # TCDecoder.ckpt
   # Wan2.1_VAE.pth
   # posi_prompt.pth
   ```

   如需复用已有权重（而不是从 ModelScope 下载），只需把上述文件放到同一目录即可，例如：

   ```bash
   mkdir -p backend/models/FlashVSR-v1.1
   cp /path/to/your/weights/* backend/models/FlashVSR-v1.1/
   ```

   FastAPI 默认从 `backend/models/FlashVSR-v1.1/posi_prompt.pth` 读取 prompt tensor；若你需要自定义路径（例如挂载到共享权重目录），可在 `backend/.env` 中设置：

   ```bash
   FLASHVSR_MODEL_PATH="/abs/path/to/FlashVSR-v1.1"
   FLASHVSR_PROMPT_TENSOR_PATH="/abs/path/to/FlashVSR-v1.1/posi_prompt.pth"
   ```
3. **复制环境变量模版并调整**
   ```bash
   # 本地开发：依赖由 docker-compose.dev.yml 提供
   cp backend/.env.dev.example backend/.env
   cp frontend/.env.example frontend/.env
   ```
   - `.env.dev.example` 预设 `localhost` 作为 PostgreSQL / Redis 主机，适用于直接运行 `fastapi dev` 的场景。
   - 如果后续需要把后端放进 Docker Compose 全栈，可改用 `cp backend/.env.compose.example backend/.env`，该模版会把主机名切换为 `postgres`、`redis`（Compose 网络内部可解析）。
   - `backend/.env` 中至少填 `DATABASE_URL`、`REDIS_URL`、`CELERY_BROKER_URL`、`CELERY_RESULT_BACKEND`，还可以调整 `MAX_CONCURRENT_TASKS`、`FLASHVSR_VERSION`、`DEFAULT_MODEL_VARIANT`（当前仅 Tiny Long 生效）。
   - `frontend/.env` 中设置 `VITE_API_BASE_URL=http://localhost:8000`（或生产后端地址）。
4. **启动依赖服务（推荐使用 Docker Compose dev）**
   ```bash
   docker compose -f docker-compose.dev.yml up -d
   ```
   然后用相同文件查看状态或日志。
5. **启动后端**
   ```bash
   source backend/.venv/bin/activate
   cd backend
   fastapi dev app/main.py
   ```
   单独在新终端运行 Celery：
   ```bash
   source backend/.venv/bin/activate
   cd backend
   celery -A app.core.celery_app worker --loglevel=info --concurrency=1
   ```
6. **启动前端**
   ```bash
   cd frontend
   pnpm dev
   ```

## Docker Compose 全栈

1. 确保宿主机安装 NVIDIA Container Toolkit 并拥有 GPU 驱动。
2. 保证模型与存储目录存在并可读写（权重仍需按上一节说明单独下发）：
   ```bash
   mkdir -p backend/models backend/storage/uploads backend/storage/results
   mkdir -p backend/models/FlashVSR-v1.1
   # 请将 FlashVSR v1.1 权重文件放入 backend/models/FlashVSR-v1.1（或在 backend/.env 中覆盖 FLASHVSR_MODEL_PATH）
   ```
3. 复制 `.env`：
   ```bash
   cp backend/.env.compose.example backend/.env
   cp frontend/.env.example frontend/.env
   ```
4. 启动堆栈：
   ```bash
   docker compose up --build
   ```
   Compose 会启动前端（`localhost:3000`）、后端（`localhost:8000`）、PostgreSQL（`localhost:5432`）与 Redis（`localhost:6379`），并挂载 `backend/models`、`backend/storage` 到容器。

### 端口与服务

- Frontend UI：`http://localhost:3000`
- Backend API：`http://localhost:8000`
- PostgreSQL：`localhost:5432`（用户/密码默认 `flashvsr`）
- Redis：`localhost:6379`

## 维护命令

- 数据库迁移：`uv --project backend run alembic upgrade head`
- 后端测试：`uv --project backend run pytest`
- 前端构建：`cd frontend && pnpm build`

## 分辨率与缩放行为说明

FlashVSR 应用的分辨率策略分为三个阶段：前端预处理、后端缩放与模型内部对齐。

- **前端预处理宽度（preprocess_width）**
  - 在上传表单中选择，例如 640 / 768 / 896 / 960 / 1024 / 1152 / 1280。
  - 后端通过 FFmpeg 将输入视频缩放到该宽度，高度按原始比例自适应（`scale=<width>:-2`），仅保证为偶数。
  - 这里不强制 128 的倍数，方便用“960 搭配 2× 超分 ≈ 1080p”这类直观选择。

- **模型前缩放与 128 倍数对齐**
  - FlashVSR（WanVideo DiT）内部自注意力窗口为 `(2, 8, 8)`，要求 VAE latent 网格维度 `(F,H,W)` 能整除窗口大小，否则会报 `Dims must divide by window size.`。
  - 为此，后端在送入模型前会：
    1. 先按 `scale`（默认 2.0）对预处理后的视频做超分；
    2. 再把结果的高宽分别向下对齐到 **128 的倍数**，并做居中裁剪。
  - 示例：预处理后为 960×540，`scale=2.0` → 理论上 1920×1080 → 实际送入模型是对齐后的 1920×1024。

- **对速度和质量的影响**
  - 计算量与分辨率面积近似成正比：从 1280 降到 960，大约是 `(960/1280)^2 ≈ 0.56`，推理速度能明显提升。
  - 高宽对齐到 128 倍数只在边缘裁掉少量像素（如 1080→1024），视觉上差异很小。
  - 在同一素材上，从 960 提高到 1024/1152 更多是增加细节恢复潜力与耗时，肉眼差异相对细微，建议按“性能预算 + 目标分辨率”选择常用档位。

**推荐组合：**
- 日常场景、接近 1080p 且兼顾速度：`preprocess_width = 960`，`scale = 2.0`。
- 追求更高分辨率（接近 2K/1440p）：`preprocess_width = 1152` 或 `1280` 搭配 2× 或 4× 超分，模型侧仍会自动对齐到最近的 128 倍数。

## 多 GPU 并行（两张 3080 20GB 示例）

FlashVSR 最简单、最稳妥的多 GPU 方案是“任务级并行”：为每张 GPU 启动一个 Celery worker，每个 worker 只见到且只使用一张卡。这样两个视频任务可并行运行；若只有一个任务，也可以在队列中同时提交多个任务以占满两张卡。

- 代码层支持：后端新增了 `FLASHVSR_DEVICE` 配置（可设为空、`cpu`、`cuda`、`cuda:0`、`cuda:1`）。为空时自动选择；设置成 `cuda` 时使用当前可见的 GPU（配合 `CUDA_VISIBLE_DEVICES` 即可把“本地 1 号卡”映射为容器/进程里的 `cuda:0`）。

### 本机（非 Docker）

在两个终端分别启动各自绑定的 worker（确保已激活 `backend/.venv`）：

```bash
# GPU0
CUDA_VISIBLE_DEVICES=0 FLASHVSR_DEVICE=cuda \
  celery -A app.core.celery_app worker --loglevel=info --concurrency=1 --max-tasks-per-child=1

# GPU1（另一个终端）
CUDA_VISIBLE_DEVICES=1 FLASHVSR_DEVICE=cuda \
  celery -A app.core.celery_app worker --loglevel=info --concurrency=1 --max-tasks-per-child=1
```

说明：将 `FLASHVSR_DEVICE` 设为 `cuda`，并用 `CUDA_VISIBLE_DEVICES` 把“物理 1 号卡/2 号卡”分别映射为进程视角下的 `cuda:0`，后端会自动识别并在日志里打印实际 GPU 名称。

### Docker Compose

把 `celery-worker` 服务复制两份，分别限定到 `device_ids: ["0"]` 与 `device_ids: ["1"]`，并设置 `CUDA_VISIBLE_DEVICES`：

```yaml
  celery-worker-0:
    extends: celery-worker
    container_name: flashvsr-celery-0
    environment:
      CUDA_VISIBLE_DEVICES: "0"
      FLASHVSR_DEVICE: cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]

  celery-worker-1:
    extends: celery-worker
    container_name: flashvsr-celery-1
    environment:
      CUDA_VISIBLE_DEVICES: "1"
      FLASHVSR_DEVICE: cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
```

注意：若保留原有的 `celery-worker`，可能会多启动一个 worker；通常建议仅启用上述两项或将原服务注释掉。

### 单视频的跨 GPU 并行（可选、高阶）

已添加“流水线并行（pipeline parallel）”的轻量支持，用于把 DiT Blocks 拆分到两张卡：

- 环境变量：
  - `FLASHVSR_PP_DEVICES`：逗号分隔设备列表，例如 `cuda:0,cuda:1` 或 `0,1`（为空表示关闭）。
  - `FLASHVSR_PP_SPLIT_BLOCK`：切分点（block 索引，以左闭右开，默认 `auto` 表示居中切分）。
  - `FLASHVSR_PP_OVERLAP_MODE`：重叠调度模式，`basic`（默认）沿用原有实现，`aggressive` 启用更激进的三阶段窗口流水线（需两卡且仅在 Tiny Long/Full 实验）。
- 行为：
  - `patch_embedding` 与前半段 blocks 放在第一个设备；后半段 blocks 与 `head` 放在最后一个设备。
  - 推理时在切分点把中间激活（和 RoPE 频率表、时间调制张量）搬迁到下一个设备。
  - Cross-Attn 的持久 KV 缓存在启用后会迁移到各自 block 所在设备。
  - 为避免设备错配，开启流水线并行时会自动禁用 VRAM management（按需上/下载）。

示例（两张 3080）

```bash
# 本机
export FLASHVSR_PP_DEVICES="0,1"        # 等价于 cuda:0,cuda:1
export FLASHVSR_PP_SPLIT_BLOCK=auto      # 或者具体层号，比如 19

# Compose 可将上述项写入 backend/.env
```

注意：默认的流水线并行以“分段驻留”为主，若想对单个长视频提速，可开启窗口级重叠（Stage0(t+1) 与 Stage1(t) 并行）：

- 额外开关：`FLASHVSR_PP_OVERLAP=1`
- 加速原理：在第 0 块设备上启动下一窗口的前半段计算（Stage0），同时第 1 块设备完成上一窗口的后半段（Stage1）。
- 适用前提：建议 `FLASHVSR_PP_DEVICES=0,1` 且足够 PCIe 带宽（无 NVLink 也可，已尽量减少跨卡搬运；RoPE/time‑mod 每窗各搬一次、激活跨卡仅一次）。

示例：

```bash
export FLASHVSR_PP_DEVICES="0,1"
export FLASHVSR_PP_SPLIT_BLOCK=auto
export FLASHVSR_PP_OVERLAP=1
export FLASHVSR_PP_OVERLAP_MODE=basic   # 或 aggressive，启用更激进的 3-stage 调度
```

若启用后希望与单卡保持一致的画质，重叠调度保持数学等价，仅调度与数据搬迁方式不同；如需最大吞吐仍建议结合“任务级并行”。

## 备注

- 确保 `backend/storage` 与 `backend/models` 在 `.gitignore` 中不会被提交，且 Docker volume 挂载保持一致。
- `FLASHVSR_CACHE_OFFLOAD` 控制流式 KV cache 的下放策略（`auto`/`cpu`/`none`，默认 `auto`）。显存 ≤ `FLASHVSR_CACHE_OFFLOAD_AUTO_THRESHOLD_GB`（默认 24GB）的 GPU 会自动把注意力缓存转存到 CPU，从而避免大分辨率视频在第 2 个窗口就 OOM；若希望始终启用或关闭，可改成 `cpu` 或 `none`。
- `FLASHVSR_STREAMING_LQ_MAX_BYTES`（默认 0，表示按分辨率自动估算）配合 `FLASHVSR_STREAMING_PREFETCH_FRAMES` 与 `FLASHVSR_STREAMING_DECODE_THREADS` 控制纯内存流式缓冲：`services/video_streaming.py` 会预先申请一个受限的环形缓冲（自动模式下会按当前分辨率与 dtype 估算，至少可容纳 `max(FLASHVSR_STREAMING_PREFETCH_FRAMES, 50)` 帧），并用多线程在 CPU 端解码 LQ 帧，达到预读阈值（默认 25 帧 ≈ 3 个 8 帧窗口 + 1 帧）后即可启动推理。显式设置 `FLASHVSR_STREAMING_LQ_MAX_BYTES`>0 时，会在不低于该自动估算值的前提下预锁定对应内存，实现 “顶满即滚动释放” 的策略。FlashVSR 推理循环通过 `release_until` 及时释放已消费的帧。
  - 该上限只影响 **输入 LQ 流** 在 CPU 内存中最多缓存多少帧，与输出分片（`FLASHVSR_CHUNKED_SAVE_*`）无直接关系；模型每次只需要几十帧的滑动窗口（约 8n+1 帧）即可连续推理。
  - 实际占用可按“目标分辨率 + 工作集帧数”估算：单帧占用近似为 `H * W * 3 * 2 bytes`（bfloat16），当前默认工作集不少于 50 帧。例如 2×1080p（模型侧约 1920×1024）时，50 帧约 0.6 GiB，因此自动模式会预留约 0.6 GiB 级别的缓冲；更高分辨率或希望更充足的预读可通过调大 `FLASHVSR_STREAMING_LQ_MAX_BYTES` 显式放宽上限。
  - 若希望小视频一次载入 CPU 内存，可将该值设得更大（如 `32GB` 或 `64GB`）；若机器内存极度紧张，也可以显式下调该值，但需注意不要低于自动估算值，否则会被自动提升并记录警告日志。
- `FLASHVSR_CHUNKED_SAVE_MIN_FRAMES` / `FLASHVSR_CHUNKED_SAVE_CHUNK_SIZE` 控制输出分片写入：当帧数超过阈值时，Celery 会把超分结果按固定帧数拆成多个 `backend/storage/tmp/chunks_*/*.mp4`，由后台进程写盘，最后再用 `ffmpeg -f concat` 合并生成最终文件。设为 0 可恢复一次性写入。
- 若推理过程中任务失败或被取消，系统会把已经写盘的分片自动合并成 `<输出文件名>_partial.mp4` 并保存在结果目录，同时在错误信息中提示路径，便于用户取走已完成的部分视频。
- `FLASHVSR_EXPORT_VIDEO_QUALITY` 控制最终结果视频在导出阶段的编码质量（整数 1–10，默认 6，数值越大质量越高、码率与文件体积越大）。该参数同时作用于普通导出与分片写盘两条路径，编码本身在 CPU 上完成，不额外占用 GPU 算力。
- 需要在 GPU 推理前先做一次 FFmpeg 采样时，可在 `backend/.env` 中设置以下变量：
  - `FFMPEG_BINARY`/`FFPROBE_BINARY`（可执行路径）
  - `PREPROCESS_FFMPEG_PRESET`（默认 `veryfast`）与 `PREPROCESS_FFMPEG_CRF`（默认 `23`），用于 CPU（libx264/libx265）编码
  - `PREPROCESS_FFMPEG_VIDEO_CODEC` 选择编码器（`libx264|h264_nvenc|libx265|hevc_nvenc`）。设置为 `h264_nvenc` 或 `hevc_nvenc` 将启用 NVENC，失败时自动回落到 CPU 编码
  - `PREPROCESS_FFMPEG_HWACCEL` 可选填写 `cuda` 开启硬件解码
  - `PREPROCESS_NVENC_PRESET`（`p1`..`p7`）、`PREPROCESS_NVENC_RC`（`vbr_hq` 等）、`PREPROCESS_NVENC_CQ`（默认 21）
前端会把“预处理策略 + 目标宽度（像素）”下发给后端：`always` 表示统一执行 `ffmpeg -vf scale=<width>:-2`，常见宽度如 640/768/896/960/1024/1152/1280。无论选择何种策略，上传 `.ts/.m2ts/.mts` 等非常见容器时后端都会自动用 FFmpeg 重新编码为 MP4 并统一像素格式为 `yuv420p`，随后沿用纯内存流式缓冲流程，并在任务结束后自动清理（包括 `pre_*.mp4` 与 `pre_audio_*.m4a` 等临时文件，避免填满 `backend/storage/tmp`）。
- Celery worker 默认并发 1，可通过 `MAX_CONCURRENT_TASKS` + Redis 锁扩展。
- 前端在生产中由 Nginx 代理 `/api` 到后端，因此无需在 `frontend/.env` 中硬编码 `VITE_API_BASE_URL`（除非使用独立部署）。
