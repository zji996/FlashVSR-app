<!-- 036d3330-baeb-4652-89d5-215bf8dbb8f9 6c876373-60d1-4060-b55b-3f0972f7c8b6 -->
# FlashVSR 全栈应用架构设计

## 技术栈

### Frontend

- **框架**: Vite + React 18 + TypeScript
- **样式**: TailwindCSS
- **状态管理**: React Query (用于服务端状态) + Zustand (用于客户端状态)
- **包管理**: pnpm

### Backend

- **框架**: FastAPI
- **任务队列**: Celery + Redis
- **ORM**: SQLAlchemy 2.0
- **包管理**: uv (使用 `--project backend` 在根目录运行)
- **模型**: FlashVSR (基于 third_party/FlashVSR)

### 数据库与缓存

- **数据库**: PostgreSQL 15
- **缓存/队列**: Redis 7

### 部署

- Docker + Docker Compose
- GPU支持 (NVIDIA Container Toolkit)

## 核心功能模块

### 1. 任务管理系统

- **任务状态**: pending, processing, completed, failed
- **进度追踪**: 实时百分比、预计剩余时间、当前处理帧数
- **任务队列**: Celery异步处理，支持并发控制

### 2. 文件存储系统

- **上传**: 支持视频文件和图像序列
- **存储**: 本地持久化 (挂载到宿主机)
- **下载**: 处理完成后的结果文件

### 3. 历史记录

- 任务列表（分页）
- 任务详情（参数、状态、结果）
- 文件管理（原始文件、处理结果）

## 项目结构

```
FlashVSR-app/
├── backend/                    # FastAPI后端
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI应用入口
│   │   ├── config.py          # 配置管理
│   │   ├── models/            # SQLAlchemy模型
│   │   ├── schemas/           # Pydantic schemas
│   │   ├── api/               # API路由
│   │   ├── services/          # 业务逻辑
│   │   ├── tasks/             # Celery任务
│   │   └── core/              # 核心工具
│   ├── pyproject.toml         # uv依赖管理
│   ├── Dockerfile
│   └── .python-version
├── frontend/                   # React前端
│   ├── src/
│   │   ├── components/        # React组件
│   │   ├── pages/             # 页面组件
│   │   ├── hooks/             # 自定义hooks
│   │   ├── api/               # API客户端
│   │   ├── stores/            # Zustand状态管理
│   │   └── types/             # TypeScript类型
│   ├── package.json
│   ├── Dockerfile
│   └── tailwind.config.js
├── docker-compose.yml          # Docker编排
├── .env.example               # 环境变量示例
├── models/                    # 模型文件目录（挂载）
└── storage/                   # 文件存储目录（挂载）
    ├── uploads/               # 上传文件
    └── results/               # 处理结果
```

## 数据库设计

### Task表

- id (UUID)
- created_at (timestamp)
- updated_at (timestamp)
- status (enum: pending, processing, completed, failed)
- input_file_path (string)
- output_file_path (string, nullable)
- parameters (jsonb): scale, sparse_ratio, local_range等
- progress (float): 0-100
- estimated_time_remaining (integer, seconds)
- error_message (text, nullable)

## API端点设计

### 任务相关

- `POST /api/tasks/` - 创建任务（上传视频）
- `GET /api/tasks/` - 获取任务列表（分页、筛选）
- `GET /api/tasks/{task_id}` - 获取任务详情
- `DELETE /api/tasks/{task_id}` - 删除任务
- `GET /api/tasks/{task_id}/progress` - 获取实时进度（WebSocket或轮询）

### 文件相关

- `GET /api/files/{task_id}/result` - 下载处理结果
- `GET /api/files/{task_id}/preview` - 预览缩略图

### 系统信息

- `GET /api/system/status` - 系统状态（队列长度、GPU使用率）

## Docker Compose服务

1. **frontend**: Nginx提供静态文件
2. **backend**: FastAPI应用（Gunicorn + Uvicorn workers）
3. **celery-worker**: Celery工作进程（GPU访问）
4. **postgres**: PostgreSQL数据库
5. **redis**: Redis缓存和消息队列

## 前端页面设计

### 主页面

- 上传区域（拖拽上传）
- 参数配置面板（scale, sparse_ratio等）
- 任务列表（实时更新）
  - 任务状态标签
  - 进度条
  - 预计剩余时间
  - 操作按钮（查看、下载、删除）

### 任务详情页

- 原始视频信息
- 处理参数
- 进度详情
- 结果预览（视频播放器）
- 下载按钮

## 关键技术实现

### 1. FlashVSR 推理封装

- 单独封装模型加载、推理和资源清理逻辑
- 实现模型单例模式（避免重复加载）
- GPU显存管理和释放策略
- 异常处理和重试机制

### 2. 进度追踪与估算

- 任务开始前解析视频元数据（总帧数、帧率、分辨率）
- Celery任务中每处理N帧更新一次进度到数据库
- 估算策略：
  - 记录已处理帧数和累计耗时
  - 计算平均每帧处理时间
  - 剩余时间 = (总帧数 - 已处理帧数) × 平均帧时间
- Frontend通过定时轮询（React Query，每2-3秒）获取进度

### 3. GPU管理

- Celery worker限制并发数为1（避免OOM）
- 使用Celery信号量或Redis锁控制同时处理的任务数
- 每个任务完成后显式清理GPU缓存

### 4. 文件清理

- 定时任务清理30天前的过期文件
- 删除任务时同步删除相关文件
- 失败任务的中间文件自动清理

## 配置要点

### Backend (uv)

- 在根目录运行: `uv --project backend <command>`
- pyproject.toml位于 `backend/pyproject.toml`

### Frontend (pnpm)

- 在frontend目录: `pnpm install`, `pnpm dev`

### Docker

- GPU支持: `deploy.resources.reservations.devices`
- 模型文件挂载: `./models:/app/models:ro`
- 存储挂载: `./storage:/app/storage`

### To-dos

- [ ] 创建backend目录结构和基础文件（pyproject.toml, main.py, config.py等）
- [ ] 定义SQLAlchemy模型（Task表）和Alembic迁移
- [ ] 实现Celery任务（FlashVSR处理逻辑、进度追踪）
- [ ] 实现FastAPI路由（任务CRUD、文件上传下载、进度查询）
- [ ] 配置frontend项目（TailwindCSS、React Query、路由）
- [ ] 实现视频上传页面和参数配置表单
- [ ] 实现任务列表和实时进度显示组件
- [ ] 实现任务详情页面和结果预览
- [ ] 创建backend Dockerfile（包含FlashVSR依赖和GPU支持）
- [ ] 创建frontend Dockerfile（多阶段构建）
- [ ] 创建docker-compose.yml（所有服务、网络、卷配置）
- [ ] 创建.env.example文件和相关配置文档