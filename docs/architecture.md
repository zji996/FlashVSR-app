# 架构设计

FlashVSR 应用围绕 Frontend、Backend 与 GPU 作业流三大层构建，同时通过 Celery/Redis 将推理负载从请求路径中摘出。

## 三层组件

- **Frontend（`frontend/`）**：Vite + React + TypeScript 负责 UI、上传、参数配置与任务状态展示；React Query 作为数据抓取层，Zustand 管理客户端筛选/分页状态；`api/` 目录封装对 `/api/*` 的调用，TailwindCSS 提供视觉体系。
- **Backend（`backend/`）**：FastAPI 提供 `/api/system/*` 和 `/api/tasks/*` 路由；SQLAlchemy/PostgreSQL 持久化任务与进度，Pydantic schemas（`app/schemas`）约束前端交互；配置集中在 `app/config`，数据库依赖于 `app/core/database.py`。
- **GPU 作业流（`services/` + `tasks/`）**：上传后由 `tasks.flashvsr_task.process_video_task` 调用 FlashVSR 推理；`services.flashvsr_service` 封装模型加载、GPU 显存管理与进度回调，Celery/Redis 负责调度与队列，`storage/` 持久化上传文件与结果。
- **LQ 流式缓冲**：`services/video_streaming.py` 维护多线程 `StreamingVideoTensor`，按照 `FLASHVSR_STREAMING_PREFETCH_FRAMES` 预读、`FLASHVSR_STREAMING_DECODE_THREADS` 个 CPU 解码线程并行处理帧。设置 `FLASHVSR_STREAMING_LQ_MAX_BYTES`>0 时会一次性预锁定同容量的环形缓冲，实现 “32GB 内存顶满就滚动释放” 的策略；值为 0 时表示无限制，只阻塞等待并不写 memmap。FlashVSR 推理循环照旧在每个 8 帧窗口结束后调用 `release_until` 释放旧帧。
- **默认模型**：`settings.DEFAULT_MODEL_VARIANT` 以及前端 `UploadForm` 默认使用 Tiny Long 版本，以便开箱就能覆盖长序列；仍可在 UI 中切换 Tiny 或 Full，或通过 `MODEL_VARIANTS_TO_PRELOAD` 预热额外 pipeline。

## 支撑设施

- **模型与存储**：FlashVSR v1.1 权重放在 `backend/models/FlashVSR-v1.1`，上传与推理结果分布在 `backend/storage/uploads` 与 `backend/storage/results`；Docker Compose 按需把这两个目录挂载到容器中。
- **异步组织**：Celery worker 限制为单线程（`MAX_CONCURRENT_TASKS=1` 默认）防止 OOM；系统状态路由会查询 `Task` 表状态计数，并调用 `FlashVSRService.inspect_assets` 来展示权重状态与就绪度。
- **运维与脚本**：`backend/tests`+`pytest` 覆盖关键接口；`docker-compose.yml`/`docker-compose.dev.yml` 可与 PostgreSQL、Redis、Celery、前端一起启动；`scripts/`（若有）可以放置迁移/维护脚本。

## 项目结构

```
FlashVSR-app/
├── backend/              # FastAPI 服务 + Celery 任务
│   ├── app/              # 路由、中间件、服务、配置
│   ├── models/           # FlashVSR 权重（不进版本控制）
│   ├── storage/          # 上传与结果
│   ├── tests/            # 后端单元/集成测试
│   └── alembic/          # 数据库迁移
├── frontend/             # Vite + React SPA
│   └── src/              # 页面、状态、API 客户端
├── docs/                 # 迁移自 README 的文档
├── docker-compose*.yml   # 本地/全栈编排
└── third_party/          # 第三方依赖或例子
```

## 关键功能亮点

1. **任务管理**：支持上传视频、校验参数与大小、记录每个任务的参数、输入/输出、进度与错误信息；后台 Celery 更新 `Task.progress` 并持久化处理帧数。
2. **FlashVSR 推理**：封装 FlashVSR 单例 Pipeline，提供 Tiny/Tiny Long/Full 变体，`settings.DEFAULT_MODEL_VARIANT` 可调整默认加载的模型。
3. **文件与清理**：上传与输出文件分别保存在 `backend/storage/uploads` 与 `backend/storage/results`，Celery Beat（或定时任务）清理 30 天以上的数据；`Task.delete` 会尝试同时清理磁盘 & DB 记录。
4. **实时状态面板**：前端在启动时会请求 `/api/system/status` 获取 GPU、任务统计与模型资产状态；必须 Guard 响应以免 React Query 在无后端时崩溃。

> 详细的 API 描述、部署流程请查看 `docs/api.md` 与 `docs/deployment.md`。
