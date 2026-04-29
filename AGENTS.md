# AGENTS.md — sam3_vs_gs2

Single git repository containing the orchestration layer and both model workspaces.

## Layout

Both workspaces follow the same **wrapper + source** pattern:

```
sam3_vs_gs2/
├── scripts/                  # orchestration (run_all.sh, infer scripts, make_comparison.py)
├── data/images/              # shared test inputs
├── notes/
├── runs/                     # generated outputs — gitignored, regenerate with run_all.sh
├── sam3_ws/                  # SAM3 uv workspace
│   ├── pyproject.toml        # wrapper project ("sam3-box"); pulls sam3 via git URL
│   ├── uv.lock
│   ├── .python-version
│   ├── .venv/
│   ├── huggingface_cache/    # gitignored; SAM3 model weights live here
│   └── sam3/                 # upstream SAM3 source (facebookresearch/sam3)
│       └── sam3/             # the importable Python package
└── gs2_ws/                   # Grounded SAM 2 uv workspace
    ├── pyproject.toml        # wrapper project ("SAM-2", package=false)
    ├── uv.lock
    ├── .python-version
    ├── .venv/
    ├── checkpoints/          # gitignored; SAM2 weights
    ├── gdino_checkpoints/    # gitignored; GroundingDINO weights
    └── source/               # upstream Grounded-SAM-2 source (IDEA-Research/Grounded-SAM-2)
        ├── sam2/             # SAM2 package + compiled _C.so
        └── grounding_dino/   # GroundingDINO package + compiled _C.so
```

**Upstream origins** are recorded in `.gitmodules` (documentation only — see [Submodule notes](#source-directories-and-gitmodules)):
- `sam3_ws/sam3` ← `https://github.com/facebookresearch/sam3.git`
- `gs2_ws/source` ← `https://github.com/IDEA-Research/Grounded-SAM-2.git`

## Main command

```bash
# From repo root
./scripts/run_all.sh IMAGE PROMPT [RUN_NAME]

# Examples
./scripts/run_all.sh data/images/truck.jpg "car. tire."
./scripts/run_all.sh ~/path/to/image.jpg "battery." my_run_name
```

- `IMAGE` is resolved to an absolute path internally (`readlink -f`), so relative or absolute paths both work.
- `RUN_NAME` defaults to `YYYY-MM-DD_HHMMSS_<stem>` when omitted.
- Logs for each step land in `runs/<run_name>/logs/sam3.log` and `logs/grounded_sam2.log`. On failure the last 80 lines are printed to stderr.

## Environment separation

| Model | Working directory | Env var override | uv invocation |
|---|---|---|---|
| SAM3 | `<repo>/sam3_ws` | `SAM3_WS` | `uv run python` |
| Grounded SAM 2 | `<repo>/gs2_ws` | `GS2_WS` | `uv run --no-sync python` |

GS2 uses `--no-sync` because its torch+CUDA packages are installed manually (not managed by the uv lockfile) to avoid triggering a CUDA extension rebuild.

GS2 additionally requires CUDA: `CUDA_HOME` defaults to `/usr/local/cuda-12.8`. Override if your CUDA is elsewhere:

```bash
CUDA_HOME=/usr/local/cuda-12.1 ./scripts/run_all.sh image.jpg "dog."
```

## Prompt format

Use dot-separated GroundingDINO-style prompts: `"car. tire."`. Both scripts receive the same raw prompt string:

- **GS2** (`run_gs2_infer.py`): passes it directly to GroundingDINO, appending a trailing `.` if missing.
- **SAM3** (`run_sam3_infer.py`): splits on `,` first, then `.`, calling the model once per concept. So `"car. tire."` → two calls: `"car"`, `"tire"`.

Do **not** pass single bare words without punctuation if you want multi-concept prompts to work in SAM3.

## SAM3 model loading order

1. `SAM3_CHECKPOINT` env var (must be a file path).
2. Scan `<repo>/sam3_ws/huggingface_cache/hub/models--facebook--sam3/snapshots/**/*.pt` — uses the most recently modified `.pt`.
3. Fall back to downloading from HuggingFace (requires internet).

## GS2 checkpoint defaults (all paths relative to `GS2_WS`)

| Asset | Default path |
|---|---|
| SAM2 checkpoint | `./checkpoints/sam2.1_hiera_large.pt` |
| SAM2 config | `source/sam2/configs/sam2.1/sam2.1_hiera_l.yaml` |
| GroundingDINO config | `source/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py` |
| GroundingDINO checkpoint | `gdino_checkpoints/groundingdino_swint_ogc.pth` |

Override any of these via CLI args to `run_gs2_infer.py` if you need a different model variant.

## Output structure

```
runs/<run_name>/
├── inputs/<image_file>
├── sam3/
│   ├── <stem>_sam3_annotated.jpg
│   ├── <stem>_sam3_mask.png
│   └── <stem>_sam3_result.json
├── grounded_sam2/
│   ├── <stem>_gs2_annotated.jpg
│   ├── <stem>_gs2_mask.png
│   └── <stem>_gs2_result.json
├── comparison/
│   ├── <stem>_side_by_side.jpg
│   ├── summary.csv          # detections + runtime for each model
│   └── notes.md             # auto-created; fill qualitative notes here
└── logs/
    ├── sam3.log
    └── grounded_sam2.log
```

`<stem>` = image filename without extension (e.g., `truck` from `truck.jpg`).

## Running steps individually

```bash
# SAM3 only
./scripts/run_sam3.sh IMAGE PROMPT OUTDIR

# GS2 only
./scripts/run_gs2.sh IMAGE PROMPT OUTDIR

# Rebuild comparison from existing run outputs (uses system python3 + Pillow only)
python3 scripts/make_comparison.py --run-dir runs/<run_name>
```

## No tests, no lint, no build

There are no test suites, linters, or build steps in this repo. Verification = running `run_all.sh` and checking that `summary.csv` is non-empty and `side_by_side.jpg` looks correct.

---

## Source directories and `.gitmodules`

`sam3_ws/sam3/` and `gs2_ws/source/` are tracked as **regular files** in this repo (not live git submodules). `.gitmodules` records the upstream URLs for reference only — `git submodule update --init` will **not** work in this state.

### Why not real submodules?

The compiled CUDA extensions (`gs2_ws/source/sam2/_C.so` and `gs2_ws/source/grounding_dino/groundingdino/_C.cpython-312-x86_64-linux-gnu.so`) are committed in-tree. Converting to real submodules requires a fresh clone of the upstream repos, which would lose these binaries and trigger a full NVCC rebuild.

### Converting to real git submodules (future)

If you want `git submodule update --init` to work, follow these steps. **Requires NVCC (`nvcc --version` to check) and ~10-20 min for CUDA builds.**

```bash
# 1. Remove source dirs from git index (keeps local files)
git rm -r --cached sam3_ws/sam3 gs2_ws/source
git commit -m "chore: deregister source dirs before submodule conversion"

# 2. Delete local copies
rm -rf sam3_ws/sam3 gs2_ws/source

# 3. Add as real submodules (clones from upstream)
git submodule add https://github.com/facebookresearch/sam3.git sam3_ws/sam3
git submodule add https://github.com/IDEA-Research/Grounded-SAM-2.git gs2_ws/source
git commit -m "chore: register upstream repos as git submodules"

# 4. Rebuild GS2 CUDA extensions
cd gs2_ws
uv pip install --no-build-isolation -e source/
# (also reinstall torch+cuda if .venv was reset)
uv pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128

# 5. Verify
cd ..
./scripts/run_all.sh data/images/truck.jpg "car. tire."
```

### Updating GS2 source from upstream

Since `gs2_ws/source/` is a regular directory, pull upstream changes manually:

```bash
cd gs2_ws/source
git init
git remote add origin https://github.com/IDEA-Research/Grounded-SAM-2.git
git fetch origin main
git merge origin/main   # or reset --hard if you want a clean sync
# Rebuild extensions if any C++ files changed
cd ..
uv pip install --no-build-isolation -e source/
```
