# AGENTS.md — sam3_vs_gs2

Single git repository containing the orchestration layer and both model workspaces.

## Layout

```
sam3_vs_gs2/
├── scripts/          # orchestration wrappers (run_all.sh, infer scripts, make_comparison.py)
├── data/images/      # shared test inputs
├── notes/
├── runs/             # generated outputs — gitignored, regenerate with run_all.sh
├── sam3_ws/          # SAM3 uv environment (pyproject.toml, uv.lock, sam3/ source)
└── gs2_ws/           # Grounded SAM 2 uv environment (pyproject.toml, uv.lock, model source)
```

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

| Model | Working directory | Env var override |
|---|---|---|
| SAM3 | `<repo>/sam3_ws` | `SAM3_WS` |
| Grounded SAM 2 | `<repo>/gs2_ws` | `GS2_WS` |

Both shells are invoked as `uv run python <script>` from their respective workspaces.

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

## GS2 checkpoint defaults (relative to `GS2_WS`)

| Asset | Default path |
|---|---|
| SAM2 checkpoint | `./checkpoints/sam2.1_hiera_large.pt` |
| SAM2 config | `configs/sam2.1/sam2.1_hiera_l.yaml` |
| GroundingDINO config | `grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py` |
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
