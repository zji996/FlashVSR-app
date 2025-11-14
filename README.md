# FlashVSR 应用

FlashVSR 构建在 FastAPI/Celery 后端与 Vite/React 前端之上，围绕 FlashVSR v1.1 推理流程提供上传、参数定制、任务监控与结果下载的端到端体验。模型权重、上传/下载文件与 GPU 作业由 Celery + Redis 统一调度，React 端界面通过 `/api` 接口实时反映任务状态。

## 文档入口

- `docs/architecture.md`：架构与项目结构视角的系统概览。
- `docs/api.md`：系统对外 OpenAPI 路由、请求与响应规范。
- `docs/deployment.md`：本地开发、Docker Compose 及运维命令的落地步骤。

## 核心亮点

- **任务进度与状态面板**：系统提供 `/api/tasks` 相关路由来创建、查询、分页和移除任务，并通过 `/api/system/status` 报告 GPU 与 FlashVSR 资产健康。
- **FlashVSR 推理封装（仅 Tiny Long）**：后端在 `backend/app/flashvsr_core` 中内聚了 FlashVSR v1.1 的推理代码，仅暴露 Tiny Long 变体，避免 third_party 代码直接参与运行时加载，同时通过 `settings.DEFAULT_MODEL_VARIANT` 自动选择默认权重。
- **上传与存储**：上传文件保存于 `backend/storage/uploads`，输出与预览放在 `backend/storage/results`；Celery 控制器在任务清理逻辑中删除磁盘数据与数据库记录。
- **FFmpeg 预处理**：前端可选择预处理宽度（如 640/768/896/960/...），GPU 推理前会无条件地用 FFmpeg 把素材重采样到该宽度（高度按原比例自适应），随后后端再按超分倍数放大并把最终高宽向下对齐到 128 的倍数，以满足 FlashVSR 的窗口约束；任务详情中会展示预估的超分输出分辨率，方便核对目标像素。

更多关于预处理宽度、缩放与 128 倍数对齐的细节，请参考 `docs/deployment.md` 中的“分辨率与缩放行为”小节。

## 快速参考

- 后端依赖：`uv --project backend sync`。
- 前端启动：`cd frontend && pnpm install && pnpm dev`。
- 启动 Celery：`source backend/.venv/bin/activate && celery -A app.core.celery_app worker --loglevel=info --concurrency=1`。
- 全栈部署：`docker compose up --build`（NVidia Container Toolkit + GPU 驱动准备）。

更多步骤请移步 `docs/deployment.md`。
