# FlashVSR 应用

FlashVSR 是一个基于 FastAPI + Celery + React 的全栈视频超分辨率处理平台。后端封装 FlashVSR 推理流程并通过 Celery 异步队列调度 GPU 工作负载，前端提供上传、参数配置、任务监控以及结果下载的完整体验。

## 架构总览

- **Frontend**：Vite + React 18 + TypeScript + TailwindCSS，使用 React Query 管理服务端状态，Zustand 管理筛选等客户端状态。
- **Backend**：FastAPI + SQLAlchemy + PostgreSQL，Celery + Redis 负责异步任务队列，封装 FlashVSR 推理及进度追踪。
- **模型与存储**：模型挂载于 `./models`，上传与结果文件持久化在 `./storage`。
- **容器编排**：Docker Compose 一键启动前端、后端、Celery Worker、PostgreSQL、Redis，并配置 NVIDIA GPU 支持。

目录结构详见项目根目录注释化示意。

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

2. 启动服务
   ```bash
   # 数据库 & Redis（可借助 Docker，或修改 .env 指向本地服务）

   # Backend API
   uv --project backend run fastapi dev app/main.py

   # Celery Worker（需要 GPU 环境）
   uv --project backend run celery -A app.core.celery_app worker --loglevel=info --concurrency=1

   # Frontend（Vite）
   cd frontend
   pnpm dev
   ```

3. 复制 `.env.example` 为 `.env` 并根据环境调整，尤其是数据库、Redis、CORS 以及默认推理参数。

### Docker Compose

1. 准备 GPU 环境（安装 NVIDIA Container Toolkit）。
2. 准备模型与存储目录：
   ```bash
   mkdir -p models storage/uploads storage/results
   ```
3. 复制 `.env.example` 为 `.env` 并根据需要覆盖默认值。
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

- `storage/.gitignore` 确保本地上传文件不被纳入版本控制。
- Docker 环境中，前端 Nginx 已配置 `/api` 代理至后端，生产构建无需额外配置 `VITE_API_BASE_URL`。
- 默认 Celery Worker 并发数为 1，以避免 FlashVSR 推理时 GPU OOM，可在 `.env` 中调节 `MAX_CONCURRENT_TASKS` 并配合 Redis 锁实现扩展。
