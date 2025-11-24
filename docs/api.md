# API 文档

所有端点挂载在 `/api` 路径下，前端在 `frontend/src/api` 有对应的 fetch helpers，后端通过 FastAPI 的 `app/api` 模块提供路由。

## 系统状态 `/api/system`

| 方法 | 路径 | 描述 |
| --- | --- | --- |
| GET | `/status` | 查询系统任务统计、GPU 信息与 FlashVSR 权重状态 |

### 响应字段

- `gpu_available`：`bool`，是否检测到可用 GPU。
- `gpu_info`：当 GPU 可用时返回 `name`、`count`、`memory_allocated`、`memory_reserved`、`memory_total`（单位：GB）。
- `tasks`：包含 `total`、`pending`、`processing`、`completed`、`failed` 各状态计数。
- `flashvsr`：模型版本、默认变体、支持/准备就绪的变体、缺失的权重文件列表、模型路径，以及 `weights_ready` 夹带布尔。

## 任务管理 `/api/tasks`

### 创建任务

- **方法**：`POST`
- **路径**：`/`
- **描述**：上传视频并创建 FlashVSR 推理任务。
- **请求体**：`multipart/form-data`
  - `file`: 视频文件（`.mp4`、`.mov`、`.avi`、`.mkv`）
  - `scale`: 超分倍数，1.0-8.0，默认 2.0。
  - `sparse_ratio`: 稀疏率，1.0-4.0，默认 2.0。
  - `local_range`: 局部范围，7-15，默认 11。
  - `seed`: 随机种子，默认 0（当前服务端始终使用 Tiny Long 变体，客户端无需选择模型）。
  - `preprocess_width`: 预处理宽度（像素），默认 640，可选常用档位 640/768/896/960/1024/1152/1280。
  - `preserve_aspect_ratio`: 是否在导出阶段按输入视频长宽比裁剪黑边恢复画面（不再对主体内容做二次缩放），布尔值，默认 `false`。
- **响应**：`TaskResponse`（见下文）。
- **错误**：400（文件/参数校验失败）、413（超出 `MAX_UPLOAD_SIZE`）或 500（保存失败）。
- **备注**：文件写入成功并排队 Celery 任务后立即返回；视频宽高/帧率等详细 `video_info` 由后台处理阶段写回。

### 列表获取

- **方法**：`GET`
- **路径**：`/`
- **查询参数**：`page`（>=1）、`page_size`（1-100）、`status`（可选 `TaskStatus`）。
- **响应**：`TaskListResponse`，包含分页信息与 `TaskResponse` 列表。

### 任务详情

- **方法**：`GET`
- **路径**：`/{task_id}`
- **描述**：获取指定任务的完整状态。
- **错误**：404（任务不存在）。

### 任务进度

- **方法**：`GET`
- **路径**：`/{task_id}/progress`
- **响应**：`TaskProgressResponse`，包含当前进度、已处理帧数、估算剩余时间等。

### 删除任务

- **方法**：`DELETE`
- **路径**：`/{task_id}`
- **描述**：移除任务记录并尝试删除上传/输出文件。
- **错误**：404（任务不存在）。

## Schema 概览

| 模式 | 说明 |
| --- | --- |
| `TaskParameters` | `scale`、`sparse_ratio`、`local_range`、`seed`、`model_variant`、`preprocess_width`、`preserve_aspect_ratio`，内置默认与字段限制。 |
| `VideoInfo` | 视频尺寸、帧数、帧率、时长、推理时间等，可选字段。 |
| `TaskResponse` | 任务元数据（`id`、`status`、`parameters`、`video_info`）、完成比例、帧计数与错误信息。 |
| `TaskProgressResponse` | 进度视图（`task_id`, `status`, `progress`, `processed_frames`, `total_frames`, `estimated_time_remaining`, `error_message`）。 |

> HTTP 示例可以通过 `backend/app/api` 中的 FastAPI docstring 或运行 `uv --project backend run fastapi dev app/main.py` 后访问 `/docs` 查看自动文档。
