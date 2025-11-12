# 部署指南

部署分为本地开发环境与 Docker Compose 全栈两条路径，关键依赖为 Python 3.11+、Node 20+、Git LFS（或手动下载权重），以及可选的 NVIDIA Container Toolkit。

## 本地开发流程

1. **安装依赖**
   ```bash
   uv --project backend sync        # 安装后端 Python 依赖
   cd frontend && pnpm install      # 安装前端依赖
   ```
1. **构建 Block-Sparse-Attention CUDA 扩展（必需）**
   ```bash
   uv pip install \
     --python backend/.venv/bin/python \
     --no-build-isolation \
     -e third_party/Block-Sparse-Attention
   ```
   该命令会在 `third_party/Block-Sparse-Attention` 中构建 `block_sparse_attn_cuda`，用于 WanVSR 模型的稀疏注意力推理；需要本地 CUDA 11.6+ 和 NVCC。
   - 建议在仓库根目录执行上述命令，以保证 `--python backend/.venv/bin/python` 指向正确；若当前目录为 `backend/`，可改用 `uv pip install --no-build-isolation -e ../third_party/Block-Sparse-Attention`（已激活虚拟环境时也可省略 `--python`）。
   - 构建过程会并发调用多个 NVCC 任务，内存占用较高。可在安装前设置并发参数来控制峰值内存：
     - 物理内存 16 GB 主机：`export MAX_JOBS=3`、`export NINJAFLAGS=-j3`
     - 物理内存 32 GB 主机：`export MAX_JOBS=6`、`export NINJAFLAGS=-j6`
     - 物理内存 32+ GB 主机：`export MAX_JOBS=8`、`export NINJAFLAGS=-j8`
     - 如仍遇到 OOM，可进一步下调上述数值至 2 或 1。
   - 如果提示 `The detected CUDA version ... mismatches ... torch.version.cuda`，需安装与当前 PyTorch (`torch.version.cuda` 显示的版本) 一致的 CUDA Toolkit，或改用匹配版本的 PyTorch wheel。
2. **准备模型权重**
   ```bash
   mkdir -p backend/models/FlashVSR-v1.1
   cd backend/models/FlashVSR-v1.1
   git clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1 tmp
   mv tmp/* . && rm -rf tmp
   ```
   或者通过 `third_party/FlashVSR/examples/...` 中的权重建立软链。
   - FastAPI 会默认从 `third_party/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth` 读取 prompt tensor（即 submodule 自带的位置）。若你更改或精简了第三方目录，可在 `backend/.env` 中设置 `FLASHVSR_PROMPT_TENSOR_PATH` 指向新的 `.pth` 文件。
3. **复制环境变量模版并调整**
   ```bash
   # 本地开发：依赖由 docker-compose.dev.yml 提供
   cp backend/.env.dev.example backend/.env
   cp frontend/.env.example frontend/.env
   ```
   - `.env.dev.example` 预设 `localhost` 作为 PostgreSQL / Redis 主机，适用于直接运行 `fastapi dev` 的场景。
   - 如果后续需要把后端放进 Docker Compose 全栈，可改用 `cp backend/.env.compose.example backend/.env`，该模版会把主机名切换为 `postgres`、`redis`（Compose 网络内部可解析）。
   - `backend/.env` 中至少填 `DATABASE_URL`、`REDIS_URL`、`CELERY_BROKER_URL`、`CELERY_RESULT_BACKEND`，还可以调整 `MAX_CONCURRENT_TASKS`、`FLASHVSR_VERSION`、`DEFAULT_MODEL_VARIANT`。
   - 可选：通过 `MODEL_VARIANTS_TO_PRELOAD`（JSON 列表）指定需要在启动时预加载的模型变体，默认只会加载 `DEFAULT_MODEL_VARIANT`。
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
2. 保证模型与存储目录存在并可读写：
   ```bash
   mkdir -p backend/models backend/storage/uploads backend/storage/results
   mkdir -p backend/models/FlashVSR-v1.1
   cd backend/models/FlashVSR-v1.1 && git lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1 tmp && mv tmp/* . && rm -rf tmp
   ```
3. 复制 `.env`：
   ```bash
   cp backend/.env.compose.example backend/.env
   cp frontend/.env.example frontend/.env
   ```
   - 若需要控制启动即加载的 pipeline，可在 `backend/.env` 中配置 `MODEL_VARIANTS_TO_PRELOAD`（例如 `["tiny","full"]`）。
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

## 备注

- 确保 `backend/storage` 与 `backend/models` 在 `.gitignore` 中不会被提交，且 Docker volume 挂载保持一致。
- Celery worker 默认并发 1，可通过 `MAX_CONCURRENT_TASKS` + Redis 锁扩展。
- 前端在生产中由 Nginx 代理 `/api` 到后端，因此无需在 `frontend/.env` 中硬编码 `VITE_API_BASE_URL`（除非使用独立部署）。
