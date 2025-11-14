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

## 分辨率与缩放行为说明

- **前端预处理宽度（preprocess_width）**  
  - 由前端表单控制，例如 640 / 768 / 896 / 960 / 1024 / 1152 / 1280。  
  - FFmpeg 会把输入视频缩放到该宽度，保持原始纵横比，高度自动调整为偶数（`scale=<width>:-2`）。  
  - 这里不强制 128 的倍数，方便用户根据目标分辨率直观选择，例如 960 搭配 2× 超分，接近 1080p。

- **模型侧对齐到 128 的倍数**  
  - FlashVSR 使用 WanVideo DiT，内部自注意力窗口是 `(2, 8, 8)`，要求 VAE latent 的网格 `(f, h, w)` 能整除窗口大小。  
  - 为避免运行时出现 `Dims must divide by window size.`，后端在送入模型前会：  
    1. 先按 `scale`（默认 2.0）放大预处理后的视频；  
    2. 再把结果的高宽分别向下对齐到 **128 的倍数**，并做居中裁剪。  
  - 示例：预处理后为 960×540，`scale=2.0` → 放大到 1920×1080，再对齐到 128 的倍数 → 实际送入模型的是 **1920×1024**，接近 1080p。

- **对速度和质量的影响**  
  - 计算量约与分辨率面积成正比，预处理宽度从 1280 降到 960，大致是 `(960/1280)^2 ≈ 0.56`，整体会明显更快。  
  - 高宽对齐到 128 的倍数只会在四周裁掉少量像素（如 1080→1024），对视觉质量影响很小，通常不需要特别关心。  
  - 在同一输入视频上，从 960 提高到 1024/1152 主要是增加细节恢复的潜力和计算量，肉眼差异相对细微，推荐按照 **性能预算 + 目标分辨率** 选择常见档位。

简单建议：  
- 日常场景、希望接近 1080p 且兼顾速度：前端选 `preprocess_width = 960`，`scale = 2.0`。  
- 追求更高分辨率（接近 2K/1440p）：可以选 1152 或 1280，再配合 2× 或 4× 超分，后端会自动对齐到最近的 128 倍数。

## 快速参考

- 后端依赖：`uv --project backend sync`。
- 前端启动：`cd frontend && pnpm install && pnpm dev`。
- 启动 Celery：`source backend/.venv/bin/activate && celery -A app.core.celery_app worker --loglevel=info --concurrency=1`。
- 全栈部署：`docker compose up --build`（NVidia Container Toolkit + GPU 驱动准备）。

更多步骤请移步 `docs/deployment.md`。
