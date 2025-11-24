#!/usr/bin/env bash
set -euo pipefail

# FlashVSR 本地开发一键启动脚本
# 参考 docs/deployment.md 的本地开发流程：
#   1) 先按文档准备 backend/.venv、模型权重、.env 等
#   2) 手动启动 dev 依赖（Postgres + Redis）：
#        docker compose -f docker-compose.dev.yml up -d
#   3) 再执行本脚本，一键启动：
#        - 后端 FastAPI dev
#        - Celery worker
#        - 前端 Vite dev server
#
# 用法：
#   bash scripts/dev.sh
#
# 脚本会：
#   - 检查 backend/.venv 是否存在，并使用其中的 fastapi/celery
#   - 简单检查 pnpm / frontend/node_modules
#   - 将三个服务以前台多进程方式启动，Ctrl+C 可一次性终止

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/backend"
FRONTEND_DIR="${ROOT_DIR}/frontend"
BACKEND_VENV="${BACKEND_DIR}/.venv"
LOG_DIR="${ROOT_DIR}/.dev-logs"

FASTAPI_BIN="${BACKEND_VENV}/bin/fastapi"
CELERY_BIN="${BACKEND_VENV}/bin/celery"

echo "[FlashVSR] 使用仓库根目录：${ROOT_DIR}"

if [ ! -x "${BACKEND_VENV}/bin/python" ]; then
  echo "[FlashVSR] 未找到 backend/.venv，请先按照 docs/deployment.md 执行后端依赖安装，例如："
  echo "  cd backend && uv venv --python 3.12 && source .venv/bin/activate && uv sync"
  exit 1
fi

if [ ! -x "$FASTAPI_BIN" ]; then
  echo "[FlashVSR] backend/.venv 中未找到 fastapi 可执行文件，请确认已在 backend/.venv 中安装 fastapi："
  echo "  cd backend && source .venv/bin/activate && uv sync"
  exit 1
fi

if [ ! -x "$CELERY_BIN" ]; then
  echo "[FlashVSR] backend/.venv 中未找到 celery 可执行文件，请确认已在 backend/.venv 中安装 Celery："
  echo "  cd backend && source .venv/bin/activate && uv sync"
  exit 1
fi

if ! command -v pnpm >/dev/null 2>&1; then
  echo "[FlashVSR] 未检测到 pnpm，请先安装 Node.js 20+ 和 pnpm，然后在 frontend/ 下执行："
  echo "  cd frontend && pnpm install"
  exit 1
fi

if [ ! -d "${FRONTEND_DIR}/node_modules" ]; then
  echo "[FlashVSR] 未检测到 frontend/node_modules，请先执行："
  echo "  cd frontend && pnpm install"
  exit 1
fi

mkdir -p "$LOG_DIR"
# 每次启动 dev 前清理上一轮留下的日志文件，避免多轮运行时日志混在一起。
if compgen -G "${LOG_DIR}/*.log" >/dev/null 2>&1; then
  echo "[FlashVSR] 清理现有 dev 日志：${LOG_DIR}/*.log"
  rm -f "${LOG_DIR}"/*.log || true
fi

echo "[FlashVSR] 如尚未启动 Postgres/Redis，请先在仓库根目录手动执行："
echo "  docker compose -f docker-compose.dev.yml up -d"
echo

PIDS=()

start_backend_api() {
  echo "[FlashVSR] 启动后端 API（fastapi dev app/main.py --host 0.0.0.0）..."
  (
    cd "$BACKEND_DIR"
    exec "$FASTAPI_BIN" dev app/main.py --host 0.0.0.0
  ) >"${LOG_DIR}/backend-api.log" 2>&1 &
  PIDS+=($!)
}

start_celery_worker() {
  echo "[FlashVSR] 启动 Celery worker..."
  (
    cd "$BACKEND_DIR"
    exec "$CELERY_BIN" -A app.core.celery_app worker --loglevel=info --concurrency=1
  ) >"${LOG_DIR}/celery-worker.log" 2>&1 &
  PIDS+=($!)
}

start_frontend() {
  echo "[FlashVSR] 启动前端 dev server（pnpm dev --host 0.0.0.0，默认端口 5173）..."
  (
    cd "$FRONTEND_DIR"
    exec pnpm dev --host 0.0.0.0
  ) >"${LOG_DIR}/frontend-dev.log" 2>&1 &
  PIDS+=($!)
}

cleanup() {
  echo
  echo "[FlashVSR] 收到退出信号，正在停止 dev 进程..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup INT TERM EXIT

start_backend_api
start_celery_worker
start_frontend

echo
echo "[FlashVSR] dev 进程已启动："
echo "  - Backend API: http://localhost:8000 （监听 0.0.0.0，可通过局域网访问）"
echo "  - Frontend UI: http://localhost:5173 （监听 0.0.0.0，可通过局域网访问）"
echo
echo "[FlashVSR] 日志文件："
echo "  - ${LOG_DIR}/backend-api.log"
echo "  - ${LOG_DIR}/celery-worker.log"
echo "  - ${LOG_DIR}/frontend-dev.log"
echo
echo "[FlashVSR] 按 Ctrl+C 可一次性停止全部 dev 进程。"

wait
