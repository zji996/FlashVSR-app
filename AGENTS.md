# Repository Guidelines

## Project Structure & Module Organization
FlashVSR keeps backend and frontend work in a single repo. `backend/app/api` hosts FastAPI routers, `core/` wires config/Celery, `services/` + `tasks/` run GPU jobs, and `schemas/` + `models/` define Pydantic/SQLAlchemy types. Frontend logic stays in `frontend/src` with fetch helpers in `api/`, routed pages in `pages/`, and state in `stores/`. Checkpoints belong in `models/`, uploads/results in `storage/uploads` + `storage/results`, and refs live in `assets/`, `docs/`, and `scripts/`.

## Build, Test, and Development Commands
- `uv --project backend sync` — install backend dependencies.
- `source backend/.venv/bin/activate && cd backend && fastapi dev app/main.py` — start the API with auto-reload (推荐方式).
- `cd backend && uv run fastapi dev app/main.py` — start the API with uv run (备选方式).
- `source backend/.venv/bin/activate && cd backend && celery -A app.core.celery_app worker --loglevel=info --concurrency=1` — run the GPU worker.
- `source backend/.venv/bin/activate && cd backend && alembic upgrade head` — apply DB migrations.
- `cd frontend && pnpm install && pnpm dev` — install and boot the Vite dev server.
- `cd frontend && pnpm lint` — lint/format the React codebase.
- `source backend/.venv/bin/activate && cd backend && pytest` — execute backend tests.
- `docker compose up --build` — launch the full stack with Postgres and Redis.

## Frontend Notes
- Tailwind CSS lives in `frontend/src/index.css` via the `@tailwind` directives and is compiled by `postcss.config.js`; if you ever need to recreate the setup, follow the Tailwind + Vite guide (create a Vite project, install `tailwindcss` plus the optional `@tailwindcss/vite` plugin, add that plugin to `vite.config.ts`, import `@tailwind base/components/utilities`, then run `pnpm dev` or `pnpm build`).
- The SPA requests `/api/system/status` on startup; when no backend is running, Vite's dev server falls back to `index.html` for those URLs, so the query resolves to an HTML string and `systemStatus.tasks` would throw. Guard the payload before reading `.tasks` (or set `VITE_API_BASE_URL` to a real backend) to avoid the blank-screen crash.

## Coding Style & Naming Conventions
Backend: follow PEP 8, 4-space indent, explicit type hints, imperative verb names for async jobs (`process_upload`), and snake_case table names. Export `router` per module, keep side effects in `services/`, schedule work via Celery tasks only. Frontend: 2-space TypeScript, PascalCase components/hooks, camelCase stores, Tailwind utility classes, and shared API types in `frontend/src/types`. Run `pnpm lint --fix` or relevant formatters before pushing.

## Testing Guidelines
Pytest plus `pytest-asyncio`/`httpx.AsyncClient` drive backend coverage; keep suites in `backend/tests/` named `test_<feature>.py` and stub Celery with `CELERY_TASK_ALWAYS_EAGER=1` to skip GPU runs. Cover upload validation, job submission, and progress polling before merging. Frontend currently relies on lint + manual smoke tests; when adding Vitest, colocate specs beside components and mock HTTP calls through the `api/` layer.

## Commit & Pull Request Guidelines
Write imperative, scope-aware commits (`backend: improve job progress polling`) capped at 72 chars and keep changesets focused. PRs should state intent, link issues, list verification commands, attach screenshots when relevant, and call out migrations, env vars, or memory impacts with accompanying `pytest`/`pnpm lint` output.

## Environment & Ops Notes
Copy `.env.example` into each service and keep secrets outside git. Place FlashVSR weights in `models/`, keep `storage/` writable or mounted under Docker, install NVIDIA Container Toolkit before `docker compose up`, and prefer symlinks inside `storage/` for large datasets.
- FlashVSR 的默认模型变体/前端默认选项均已切换为 Tiny Long；如需复用 Tiny 或 Full，请在 `backend/.env` 的 `DEFAULT_MODEL_VARIANT` / `MODEL_VARIANTS_TO_PRELOAD`（以及前端选择器）中显式调整。
- `FLASHVSR_STREAMING_LQ_MAX_BYTES`（默认 0 表示不限）+ `FLASHVSR_STREAMING_PREFETCH_FRAMES` 控制纯内存 LQ 流式缓冲：后台线程按预读阈值（至少 8n+1 帧）填充，推理线程在每个 8 帧窗口结束后调用 `release_until` 即刻释放旧帧，全程不写 `*.memmap`。想让小视频一次性载入 CPU，可把该阈值调成更大的值，例如 `64GB`。
- 输出帧数超过 `FLASHVSR_CHUNKED_SAVE_MIN_FRAMES` 时，会按 `FLASHVSR_CHUNKED_SAVE_CHUNK_SIZE` 帧拆分写入 `backend/storage/tmp/chunks_*`，异步落盘后使用 FFmpeg concat 合并；把阈值设为 0 即可恢复一次性写文件。
- 上传 `.ts`/`.m2ts`/`.mts`/其他非常见后缀时，即便 `preprocess_strategy=none`，后端也会自动用 FFmpeg 重新编码成 MP4 确保 imageio/FlashVSR 可读；若再配合 `preprocess_width`，会同时执行缩放。
- 若需要在 GPU 推理前先用 FFmpeg 对素材降采样，前端的“预处理策略 + 预处理宽度”会被写入 `TaskParameters`。`preprocess_strategy=always` 时 Celery 会无条件执行 `ffmpeg -vf scale=<width>:-2` 并把临时文件写入 `backend/storage/tmp`；`none` 则跳过缩放但仍可触发上面的容器规范化。FFmpeg/ffprobe 路径通过 `FFMPEG_BINARY` / `FFPROBE_BINARY` 配置，可在 `.env` 中连同 `PREPROCESS_FFMPEG_PRESET`、`PREPROCESS_FFMPEG_CRF` 一起调整，且整个流程仍与纯内存流式缓冲兼容。
- 留意 `FLASHVSR_CACHE_OFFLOAD`（`auto`/`cpu`/`none`），默认在 ≤24 GB GPU 上把 WanVideo 的 KV cache 下放到 CPU，确保 2×~3× 超分也能稳定运行；可在 `.env` 中覆盖。
