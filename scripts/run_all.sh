#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 IMAGE PROMPT [RUN_NAME]" >&2
  exit 2
fi
IMAGE="$(readlink -f "$1")"
PROMPT="$2"
RUN_NAME="${3:-$(date +%Y-%m-%d_%H%M%S)_$(basename "${IMAGE%.*}")}" 
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/runs/$RUN_NAME"
mkdir -p "$RUN_DIR/inputs" "$RUN_DIR/sam3" "$RUN_DIR/grounded_sam2" "$RUN_DIR/comparison" "$RUN_DIR/logs"
cp "$IMAGE" "$RUN_DIR/inputs/"

run_step() {
  local name="$1"
  shift
  echo "[$name] running..."
  if ! "$@" > "$RUN_DIR/logs/${name}.log" 2>&1; then
    echo "[$name] failed. Last log lines:" >&2
    tail -80 "$RUN_DIR/logs/${name}.log" >&2 || true
    exit 1
  fi
  echo "[$name] done. Log: $RUN_DIR/logs/${name}.log"
}

run_step sam3 "$ROOT/scripts/run_sam3.sh" "$IMAGE" "$PROMPT" "$RUN_DIR/sam3"
run_step grounded_sam2 "$ROOT/scripts/run_gs2.sh" "$IMAGE" "$PROMPT" "$RUN_DIR/grounded_sam2"
python3 "$ROOT/scripts/make_comparison.py" --run-dir "$RUN_DIR"
cat "$RUN_DIR/comparison/summary.csv"
echo "Run complete: $RUN_DIR"
