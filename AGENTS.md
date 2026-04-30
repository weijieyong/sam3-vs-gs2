# AGENTS.md — sam3_vs_gs2

Two isolated model workspaces (SAM3 + Grounded SAM 2) compared side-by-side. No tests, no lint, no build steps. Verification = run `run_all.sh` and check `summary.csv` is non-empty.

## Layout

```
sam3_vs_gs2/
├── scripts/              # run_all.sh, run_sam3.sh, run_gs2.sh, run_*_infer.py, make_comparison.py
├── data/images/          # shared test inputs
├── docs/                 # static assets (README images)
├── runs/                 # gitignored — generated outputs
├── sam3_ws/              # SAM3 uv workspace
│   ├── pyproject.toml
│   ├── huggingface_cache/    # gitignored; SAM3 weights
│   └── sam3/sam3/            # importable SAM3 package
└── gs2_ws/               # Grounded SAM 2 uv workspace
    ├── pyproject.toml
    ├── checkpoints/          # gitignored; SAM2 weights
    ├── gdino_checkpoints/    # gitignored; GroundingDINO weights
    └── source/               # Grounded-SAM-2 upstream source
        ├── sam2/             # SAM2 package + compiled _C.so
        └── grounding_dino/   # GroundingDINO package + compiled _C.so
```

## Commands

```bash
# Full pipeline (IMAGE can be relative or absolute)
./scripts/run_all.sh IMAGE PROMPT [RUN_NAME]

# Individual steps
./scripts/run_sam3.sh IMAGE PROMPT OUTDIR
./scripts/run_gs2.sh  IMAGE PROMPT OUTDIR

# Rebuild comparison only (no inference)
python3 scripts/make_comparison.py --run-dir runs/<run_name>
```

- `RUN_NAME` defaults to `YYYY-MM-DD_HHMMSS_<stem>` when omitted.
- Logs: `runs/<run_name>/logs/{sam3,grounded_sam2}.log` — last 80 lines printed to stderr on failure.

## Environment

| Model | Working dir | Env var override | Invocation |
|---|---|---|---|
| SAM3 | `sam3_ws/` | `SAM3_WS` | `uv run python` |
| Grounded SAM 2 | `gs2_ws/` | `GS2_WS` | `uv run python` |

- `CUDA_HOME` defaults to `/usr/local/cuda-12.8`. Override if different: `CUDA_HOME=/usr/local/cuda-12.1 ./scripts/run_all.sh ...`
- Both workspaces are fully uv-managed. `uv sync` installs all deps and builds CUDA extensions.

## Prompt Format

Use dot-separated GroundingDINO-style: `"car. tire."` — **not bare words**.

- **GS2** (`run_gs2_infer.py`): passes prompt directly to GroundingDINO; appends trailing `.` if missing.
- **SAM3** (`run_sam3_infer.py`): splits on `,` then `.` → one inference call per concept (`"car. tire."` → `"car"`, `"tire"`).

## Model Checkpoints

**SAM3** — loading order:
1. `SAM3_CHECKPOINT` env var (file path)
2. `sam3_ws/huggingface_cache/hub/models--facebook--sam3/snapshots/**/*.pt` (most recently modified)
3. Auto-download from HuggingFace (requires internet)

**GS2** — defaults (paths relative to `gs2_ws/`):

| Asset | Default path |
|---|---|
| SAM2 checkpoint | `checkpoints/sam2.1_hiera_large.pt` |
| SAM2 config | `configs/sam2.1/sam2.1_hiera_l.yaml` ⚠️ Hydra package-relative path — NOT a filesystem path |
| GroundingDINO config | `source/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py` |
| GroundingDINO checkpoint | `gdino_checkpoints/groundingdino_swint_ogc.pth` |

Override via CLI args to `run_gs2_infer.py`.

## Output Structure

```
runs/<run_name>/
├── inputs/           # copy of input image
├── sam3/             # <stem>_sam3_{annotated.jpg,mask.png,result.json}
├── grounded_sam2/    # <stem>_gs2_{annotated.jpg,mask.png,result.json}
├── comparison/       # <stem>_side_by_side.jpg, summary.csv, notes.md
└── logs/             # sam3.log, grounded_sam2.log
```

`<stem>` = image filename without extension (e.g. `truck` from `truck.jpg`).

## Constraints

- **NEVER** run `git submodule update --init` — `sam3_ws/sam3/` and `gs2_ws/source/` are regular tracked files, not live submodules.
- **NEVER** edit `gs2_ws/source/sam2/_C.so` or `gs2_ws/source/grounding_dino/groundingdino/_C.cpython-312-x86_64-linux-gnu.so` — compiled CUDA extensions committed in-tree.
- **NEVER** commit `huggingface_cache/`, `checkpoints/`, `gdino_checkpoints/`, or `.venv/` — all gitignored.
- **DO NOT** edit files under `runs/` — always regenerate via `run_all.sh`.
