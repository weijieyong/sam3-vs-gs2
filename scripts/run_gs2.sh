#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 IMAGE PROMPT OUTDIR" >&2
  exit 2
fi
IMAGE="$1"
PROMPT="$2"
OUTDIR="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GS2_WS="${GS2_WS:-$(cd "$SCRIPT_DIR/.." && pwd)/gs2_ws}"
cd "$GS2_WS"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
EXTRA_ARGS=()
[ -n "${GS2_BOX_THRESH:-}" ]  && EXTRA_ARGS+=(--box-threshold  "$GS2_BOX_THRESH")
[ -n "${GS2_TEXT_THRESH:-}" ] && EXTRA_ARGS+=(--text-threshold "$GS2_TEXT_THRESH")
uv run python "$SCRIPT_DIR/run_gs2_infer.py" --image "$IMAGE" --prompt "$PROMPT" --output-dir "$OUTDIR" "${EXTRA_ARGS[@]}"
