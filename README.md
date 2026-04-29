# SAM3 vs Grounded SAM 2 Comparison Workspace

This workspace compares SAM3 and Grounded SAM 2 without merging their Python environments.

## Layout

```text
sam3_vs_gs2/
├── data/images/              # shared test inputs
├── scripts/                  # orchestration and wrapper scripts
└── runs/<run_name>/           # consolidated outputs per run
    ├── inputs/
    ├── sam3/
    ├── grounded_sam2/
    └── comparison/
```

## Run one comparison

```bash
cd ~/03_Exp/sam3_vs_gs2
./scripts/run_all.sh data/images/example.jpg "battery."
```

Or use an absolute input path:

```bash
./scripts/run_all.sh ~/03_Exp/grounded_sam2_ws/source/Grounded-SAM-2/notebooks/images/truck.jpg "car. tire."
```

## Environment separation

- SAM3 runs from `~/03_Exp/sam3_ws` using its own `uv` environment.
- Grounded SAM 2 runs from `~/03_Exp/grounded_sam2_ws/source/Grounded-SAM-2` using its own `uv` environment.
- This workspace only orchestrates runs and consolidates outputs.

Override paths if needed:

```bash
SAM3_WS=/path/to/sam3_ws GS2_WS=/path/to/Grounded-SAM-2 ./scripts/run_all.sh image.jpg "prompt."
```

## Outputs

Each run produces:

- `sam3/*_annotated.jpg`
- `sam3/*_mask.png`
- `sam3/*_result.json`
- `grounded_sam2/*_annotated.jpg`
- `grounded_sam2/*_mask.png`
- `grounded_sam2/*_result.json`
- `comparison/*_side_by_side.jpg`
- `comparison/summary.csv`
- `comparison/notes.md`
