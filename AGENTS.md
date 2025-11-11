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
