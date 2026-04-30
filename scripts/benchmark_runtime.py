#!/usr/bin/env python3
"""Benchmark SAM3 vs GS2 inference runtime.

- N runs per image, alternating run order to cancel GPU boost-clock bias
- Reports per-image medians and overall median
- Appends results to notes/runtime_comparison.md

Usage:
    uv run python scripts/benchmark_runtime.py
    uv run python scripts/benchmark_runtime.py --n 3 --prompt "objects."
    uv run python scripts/benchmark_runtime.py --sam3-conf-thresh 0.2 --gs2-box-thresh 0.2 --gs2-text-thresh 0.2
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
NOTES_FILE = ROOT / "notes" / "runtime_comparison.md"


def run_infer(
    shell_script: Path, image: Path, prompt: str, env_overrides: dict
) -> float:
    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env.update(env_overrides)
        proc = subprocess.run(
            [str(shell_script), str(image), prompt, tmpdir],
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{shell_script.name} failed on {image.name}:\n{proc.stderr[-1000:]}"
            )
        jsons = list(Path(tmpdir).glob("*_result.json"))
        if not jsons:
            raise RuntimeError(
                f"No result JSON found in {tmpdir}.\nstdout:\n{proc.stdout[:500]}"
            )
        return json.loads(jsons[0].read_text())["runtime_sec"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark SAM3 vs GS2 inference runtime")
    ap.add_argument(
        "--n", type=int, default=5, help="Runs per image per model (default: 5)"
    )
    ap.add_argument(
        "--prompt", default="objects.", help="Text prompt (default: 'objects.')"
    )
    ap.add_argument(
        "--images-dir",
        default=str(ROOT / "data" / "images"),
        help="Directory of input images",
    )
    ap.add_argument("--sam3-conf-thresh", type=float, default=0.5)
    ap.add_argument("--gs2-box-thresh", type=float, default=0.35)
    ap.add_argument("--gs2-text-thresh", type=float, default=0.25)
    args = ap.parse_args()

    images = sorted(
        list(Path(args.images_dir).glob("*.jpg"))
        + list(Path(args.images_dir).glob("*.png"))
    )
    if not images:
        print(f"No images found in {args.images_dir}", file=sys.stderr)
        return 1

    sam3_script = SCRIPT_DIR / "run_sam3.sh"
    gs2_script = SCRIPT_DIR / "run_gs2.sh"
    env_sam3 = {"SAM3_CONF_THRESH": str(args.sam3_conf_thresh)}
    env_gs2 = {
        "GS2_BOX_THRESH": str(args.gs2_box_thresh),
        "GS2_TEXT_THRESH": str(args.gs2_text_thresh),
    }

    per_image: list[dict] = []
    all_sam3_times: list[float] = []
    all_gs2_times: list[float] = []

    for image in images:
        print(f"\n[{image.name}] {args.n} runs each, alternating order", flush=True)
        sam3_times: list[float] = []
        gs2_times: list[float] = []

        for i in range(args.n):
            label = f"  run {i + 1}/{args.n}"
            if i % 2 == 0:  # even runs: SAM3 first
                t = run_infer(sam3_script, image, args.prompt, env_sam3)
                sam3_times.append(t)
                print(f"{label}  SAM3={t:.3f}s", end="  ", flush=True)
                t = run_infer(gs2_script, image, args.prompt, env_gs2)
                gs2_times.append(t)
                print(f"GS2={t:.3f}s", flush=True)
            else:  # odd runs: GS2 first
                t = run_infer(gs2_script, image, args.prompt, env_gs2)
                gs2_times.append(t)
                print(f"{label}  GS2={t:.3f}s", end="  ", flush=True)
                t = run_infer(sam3_script, image, args.prompt, env_sam3)
                sam3_times.append(t)
                print(f"SAM3={t:.3f}s", flush=True)

        sam3_med = statistics.median(sam3_times)
        gs2_med = statistics.median(gs2_times)
        ratio = gs2_med / sam3_med if sam3_med else float("inf")
        all_sam3_times.extend(sam3_times)
        all_gs2_times.extend(gs2_times)
        per_image.append(
            {
                "image": image.name,
                "sam3_median": sam3_med,
                "gs2_median": gs2_med,
                "ratio": ratio,
                "sam3_all": sam3_times,
                "gs2_all": gs2_times,
            }
        )
        print(f"  → median  SAM3={sam3_med:.3f}s  GS2={gs2_med:.3f}s  ({ratio:.2f}x)")

    overall_sam3 = statistics.median(all_sam3_times)
    overall_gs2 = statistics.median(all_gs2_times)
    overall_ratio = overall_gs2 / overall_sam3 if overall_sam3 else float("inf")
    print(f"\n{'=' * 50}")
    print(
        f"Overall median  SAM3={overall_sam3:.3f}s  GS2={overall_gs2:.3f}s  ({overall_ratio:.2f}x)"
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    table_rows = "\n".join(
        f"| `{r['image']}` | {r['sam3_median']:.3f}s | {r['gs2_median']:.3f}s | {r['ratio']:.2f}x |"
        for r in per_image
    )
    raw_lines = "\n".join(
        f"- `{r['image']}`: SAM3=[{', '.join(f'{t:.3f}' for t in r['sam3_all'])}]"
        f"  GS2=[{', '.join(f'{t:.3f}' for t in r['gs2_all'])}]"
        for r in per_image
    )
    section = f"""
---

## N-Runs Median Benchmark — {timestamp}

- **N**: {args.n} runs per image per model
- **Prompt**: `"{args.prompt}"`
- **Thresholds**: `SAM3_CONF_THRESH={args.sam3_conf_thresh}`, `GS2_BOX_THRESH={args.gs2_box_thresh}`, `GS2_TEXT_THRESH={args.gs2_text_thresh}`
- **Order**: alternated per run (even=SAM3 first, odd=GS2 first) to cancel GPU boost-clock bias
- **Images**: {len(images)} ({", ".join(im.name for im in images)})

| Image | SAM3 median | GS2 median | GS2/SAM3 ratio |
|---|---|---|---|
{table_rows}
| **Overall** | **{overall_sam3:.3f}s** | **{overall_gs2:.3f}s** | **{overall_ratio:.2f}x** |

Raw per-image times:

{raw_lines}
"""
    with open(NOTES_FILE, "a") as f:
        f.write(section)
    print(f"\nResults appended to {NOTES_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
