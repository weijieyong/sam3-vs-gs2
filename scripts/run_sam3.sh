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
SAM3_WS="${SAM3_WS:-$(cd "$SCRIPT_DIR/.." && pwd)/sam3_ws}"
cd "$SAM3_WS"
EXTRA_ARGS=()
[ -n "${SAM3_CONF_THRESH:-}" ] && EXTRA_ARGS+=(--confidence-threshold "$SAM3_CONF_THRESH")
uv run python "$SCRIPT_DIR/run_sam3_infer.py" --image "$IMAGE" --prompt "$PROMPT" --output-dir "$OUTDIR" "${EXTRA_ARGS[@]}"
