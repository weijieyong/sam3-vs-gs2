#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=== Grounded-SAM-2 dev environment setup ==="

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

if ! command -v nvcc &>/dev/null && [ -z "${CUDA_HOME:-}" ]; then
    echo "WARNING: nvcc not found and CUDA_HOME is unset."
    echo "  SAM-2 CUDA extension and GroundingDINO C extension will not compile."
    echo "  Set CUDA_HOME before running, e.g.:"
    echo "    export CUDA_HOME=/usr/local/cuda-12.8"
    echo "  Or skip the CUDA extension with: export SAM2_BUILD_CUDA=0"
    echo ""
fi

echo "[1/5] Pinning Python 3.12 ..."
uv python pin 3.12

echo "[2/5] Syncing base deps (keeps manually installed torch wheels, no project install) ..."
uv sync --no-build-isolation --no-install-project --inexact

echo "[3/5] Checking PyTorch availability ..."
if ! uv run python -c "import torch, torchvision" >/dev/null 2>&1; then
    echo ""
    echo "ERROR: PyTorch not found in .venv."
    echo "Install the CUDA-specific wheel first, then rerun this script:"
    echo ""
    echo "  source .venv/bin/activate"
    echo "  # CUDA 12.8 (this machine):"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
    echo "  # CUDA 12.4 fallback:"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    echo ""
    exit 2
fi

echo "[4/5] Installing SAM-2 package (editable, with CUDA extension) ..."
uv pip install --no-build-isolation -e .

echo "[5/5] Installing GroundingDINO (editable, requires nvcc for Deformable Attention) ..."
uv pip install --no-build-isolation -e grounding_dino

echo ""
echo "=== Verifying installs ==="
uv run python -c "
import torch
import sam2
import sam2._C
import groundingdino
print('  torch          :', torch.__version__)
print('  cuda available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  device         :', torch.cuda.get_device_name(0))
print('  sam2           : OK')
print('  sam2._C (ext)  : OK')
print('  groundingdino  : OK')
" && echo "" && echo "=== Setup complete ===" || echo "WARNING: verification step failed - check output above"

echo ""
echo "Usage:"
echo "  uv run python grounded_sam2_hf_model_demo.py"
echo "  uv run python grounded_sam2_tracking_demo.py"
echo ""
echo "Download model checkpoints when ready:"
echo "  cd checkpoints && bash download_ckpts.sh && cd .."
echo "  cd gdino_checkpoints && bash download_ckpts.sh && cd .."
