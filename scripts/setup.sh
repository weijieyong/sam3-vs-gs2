#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"

echo "[root] uv sync (comparison scripts)..."
(cd "$REPO" && uv sync)

echo "[sam3] uv sync..."
(cd "$REPO/sam3_ws" && uv sync)

echo "[gs2] uv sync (builds CUDA extensions, ~1-2 min)..."
(cd "$REPO/gs2_ws" && uv sync)

echo "Done. Run: ./scripts/run_all.sh <image> <prompt>"
