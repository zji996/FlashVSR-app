#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash scripts/install_block_sparse_attn.sh        # 常规安装（若已安装则跳过）
#   bash scripts/install_block_sparse_attn.sh --force  # 强制重新构建扩展
#
# 脚本特性：
#   - 自动定位仓库根目录并使用 backend/.venv 里的 Python
#   - 若 backend/.venv 中已能导入 block_sparse_attn，则默认跳过编译，避免重复高负载构建
#   - 若未设置 MAX_JOBS/NINJAFLAGS，则按物理内存自动选择较保守的并发度，降低 OOM 风险

# Resolve repo root (directory that contains this script's parent)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="$ROOT_DIR/backend/.venv/bin/python"

cd "$ROOT_DIR"

if [ ! -x "$PY_BIN" ]; then
  echo "[FlashVSR] backend/.venv 不存在，请先执行："
  echo "  uv --project backend sync"
  exit 1
fi

FORCE_REBUILD="${1-}"

if [ "$FORCE_REBUILD" != "--force" ]; then
  echo "[FlashVSR] 检查 backend/.venv 中是否已安装 block_sparse_attn..."
  if "$PY_BIN" - << 'PY'
import importlib

try:
    importlib.import_module("block_sparse_attn")
except Exception:
    raise SystemExit(1)

print("ok")
PY
  then
    echo "[FlashVSR] 已检测到 block_sparse_attn，可用，跳过构建。"
    echo "[FlashVSR] 如需强制重新编译，可运行：bash scripts/install_block_sparse_attn.sh --force"
    exit 0
  else
    echo "[FlashVSR] backend/.venv 中尚未安装 block_sparse_attn，开始构建..."
  fi
else
  echo "[FlashVSR] 强制重新构建 Block-Sparse-Attention 扩展..."
fi

# 若用户未显式设置并发参数，则按物理内存设置相对保守的默认值
if [ -z "${MAX_JOBS:-}" ] || [ -z "${NINJAFLAGS:-}" ]; then
  if [ -r /proc/meminfo ]; then
    MEM_KB=$(grep -i '^MemTotal:' /proc/meminfo | awk '{print $2}')
    MEM_GB=$(( MEM_KB / 1024 / 1024 ))
  else
    MEM_GB=16
  fi

  if [ "$MEM_GB" -le 20 ]; then
    export MAX_JOBS="${MAX_JOBS:-3}"
    export NINJAFLAGS="${NINJAFLAGS:--j3}"
  elif [ "$MEM_GB" -le 36 ]; then
    export MAX_JOBS="${MAX_JOBS:-6}"
    export NINJAFLAGS="${NINJAFLAGS:--j6}"
  else
    export MAX_JOBS="${MAX_JOBS:-8}"
    export NINJAFLAGS="${NINJAFLAGS:--j8}"
  fi

  echo "[FlashVSR] 检测到物理内存约 ${MEM_GB}GB，使用并发配置：MAX_JOBS=${MAX_JOBS} NINJAFLAGS=${NINJAFLAGS}"
else
  echo "[FlashVSR] 使用用户自定义并发配置：MAX_JOBS=${MAX_JOBS} NINJAFLAGS=${NINJAFLAGS}"
fi

echo "[FlashVSR] 在 backend/.venv 中构建 Block-Sparse-Attention 扩展（非 editable，便于复用已构建 wheel）..."
uv pip install \
  --python "$PY_BIN" \
  --no-build-isolation \
  "$ROOT_DIR/third_party/Block-Sparse-Attention"

echo "[FlashVSR] Block-Sparse-Attention 安装完成。"
