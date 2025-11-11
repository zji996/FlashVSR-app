# FlashVSR 应用

FlashVSR 是一个基于 FastAPI + Celery + React 的全栈视频超分辨率处理平台。后端封装 FlashVSR 推理流程并通过 Celery 异步队列调度 GPU 工作负载，前端提供上传、参数配置、任务监控以及结果下载的完整体验。

## 架构总览

- **Frontend**：Vite + React 18 + TypeScript + TailwindCSS，使用 React Query 管理服务端状态，Zustand 管理筛选等客户端状态。
- **Backend**：FastAPI + SQLAlchemy + PostgreSQL，Celery + Redis 负责异步任务队列，封装 FlashVSR 推理及进度追踪。
- **模型与存储**：FlashVSR v1.1 权重位于 `backend/models/FlashVSR-v1.1`，上传与结果文件持久化在 `backend/storage`。
- **容器编排**：Docker Compose 一键启动前端、后端、Celery Worker、PostgreSQL、Redis，并配置 NVIDIA GPU 支持。

## 项目结构

```
FlashVSR-app/
├── backend/              # 后端服务（自包含）
│   ├── .env             # 后端环境变量
│   ├── .env.example     # 后端环境变量模板
│   ├── app/             # FastAPI 应用
│   ├── models/          # AI 模型权重
│   ├── storage/         # 上传和结果存储
│   ├── tests/           # 后端测试
│   └── alembic/         # 数据库迁移
├── frontend/            # 前端服务（自包含）
│   ├── .env             # 前端环境变量
│   ├── .env.example     # 前端环境变量模板
│   └── src/             # React 源码
├── third_party/         # 第三方代码
├── docs/                # 项目文档
└── docker-compose.yml   # 容器编排
```

## 快速开始

### 本地开发（不使用 Docker）

1. 安装依赖
   ```bash
   # Backend（需要 Python 3.11+）
   uv --project backend sync

   # Frontend（需要 Node.js 20+，pnpm）
   cd frontend
   pnpm install
   ```

2. 准备 FlashVSR v1.1 权重（首次运行必需）
   ```bash
   mkdir -p backend/models/FlashVSR-v1.1
   cd backend/models/FlashVSR-v1.1
   git clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1 tmp
   mv tmp/* . && rm -rf tmp
   ls
   # 需包含 diffusion_pytorch_model_streaming_dmd.safetensors \
   #        LQ_proj_in.ckpt TCDecoder.ckpt Wan2.1_VAE.pth
   ```
   或者复用 `third_party/FlashVSR/examples/WanVSR/FlashVSR-v1.1` 下的权重并建软链接到 `backend/models/FlashVSR-v1.1`。

3. 配置环境变量
   
   **后端配置：**
   ```bash
   cp backend/.env.example backend/.env
   ```
   
   编辑 `backend/.env` 确保以下配置正确：
   ```env
   DATABASE_URL=postgresql+psycopg2://flashvsr:flashvsr@localhost:5432/flashvsr
   REDIS_URL=redis://localhost:6379/0
   CELERY_BROKER_URL=redis://localhost:6379/1
   CELERY_RESULT_BACKEND=redis://localhost:6379/2
   ```
   
   **前端配置：**
   ```bash
   cp frontend/.env.example frontend/.env
   ```
   
   编辑 `frontend/.env` 设置后端 API 地址：
   ```env
   VITE_API_BASE_URL=http://localhost:8000
   ```

4. 启动数据库与 Redis（使用 Docker）
   ```bash
   # 启动 PostgreSQL 和 Redis（后台运行）
   docker compose -f docker-compose.dev.yml up -d
   
   # 查看服务状态
   docker compose -f docker-compose.dev.yml ps
   
   # 查看日志
   docker compose -f docker-compose.dev.yml logs -f
   
   # 停止服务（不删除数据）
   docker compose -f docker-compose.dev.yml stop
   
   # 停止并删除容器（保留数据卷）
   docker compose -f docker-compose.dev.yml down
   ```

5. 启动后端服务
   
   **方式一：激活虚拟环境后运行（推荐）**
   ```bash
   # 激活 backend 虚拟环境
   source backend/.venv/bin/activate
   
   # 启动 Backend API（在 backend 目录下执行）
   cd backend
   fastapi dev app/main.py
   
   # Celery Worker（需要 GPU 环境，新开终端）
   source backend/.venv/bin/activate
   cd backend
   celery -A app.core.celery_app worker --loglevel=info --concurrency=1
   ```
   
   **方式二：使用 uv run 直接运行**
   ```bash
   # Backend API（在 backend 目录下执行）
   cd backend && uv run fastapi dev app/main.py
   
   # Celery Worker（需要 GPU 环境，新开终端）
   cd backend && uv run celery -A app.core.celery_app worker --loglevel=info --concurrency=1
   ```

6. 启动前端服务
   ```bash
   # Frontend（Vite）
   cd frontend
   pnpm dev
   ```

### Docker Compose

1. 准备 GPU 环境（安装 NVIDIA Container Toolkit）。
2. 准备模型与存储目录：
   ```bash
   mkdir -p backend/models backend/storage/uploads backend/storage/results
   mkdir -p backend/models/FlashVSR-v1.1
   cd backend/models/FlashVSR-v1.1 && git lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1 tmp \
     && mv tmp/* . && rm -rf tmp
   ```
3. 配置环境变量：
   ```bash
   cp backend/.env.example backend/.env
   cp frontend/.env.example frontend/.env
   ```
   根据需要修改配置。Compose 会将宿主机的 `./backend/models` 与 `./backend/storage` 挂载到容器的 `/app/models`、`/app/storage`。
4. 构建并启动：
   ```bash
   docker compose up --build
   ```

服务端口：

- Frontend：`http://localhost:3000`
- Backend：`http://localhost:8000`
- PostgreSQL：`localhost:5432`（用户名/密码默认 `flashvsr`）
- Redis：`localhost:6379`

Celery Beat 会每天凌晨 3 点自动清理超出保留期的旧任务及其文件。

## 关键功能

- 任务创建与上传校验（类型、大小、参数合法性）。
- 视频元数据解析（分辨率 / 帧率 / 帧数 / 时长）以及实时进度更新（帧计数与耗时估算）。
- FlashVSR 推理封装（单例 Pipeline、GPU 显存管理、进度回调、结果合并）。
- 支持 FlashVSR v1.1 Tiny / Tiny Long / Full 变体选择，系统状态面板实时提示权重是否就绪。
- 文件管理（原始文件、结果文件、JPG 预览）与 30 天过期清理任务。
- 前端实时任务列表、进度条、结果预览与下载。

## 常用命令

```bash
# 数据迁移
uv --project backend run alembic upgrade head

# 运行后端测试（若添加）
uv --project backend run pytest

# 前端构建
cd frontend
pnpm build
```

## 备注

- `backend/storage/` 和 `backend/models/` 已在 `.gitignore` 中忽略，本地上传文件和模型不被纳入版本控制。
- Docker 环境中，前端 Nginx 已配置 `/api` 代理至后端，生产构建无需额外配置 `VITE_API_BASE_URL`。
- 默认 Celery Worker 并发数为 1，以避免 FlashVSR 推理时 GPU OOM，可在 `backend/.env` 中调节 `MAX_CONCURRENT_TASKS` 并配合 Redis 锁实现扩展。
- `backend/.env` 提供 `FLASHVSR_VERSION`、`FLASHVSR_MODEL_PATH`、`DEFAULT_MODEL_VARIANT` 等配置，可用于切换不同的 FlashVSR 权重或默认变体。
- 前后端分离配置：`backend/.env` 管理后端配置，`frontend/.env` 管理前端配置，便于独立部署和维护。