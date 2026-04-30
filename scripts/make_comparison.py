#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


def _load_font(size):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def make_side_by_side(paths, labels, out_path):
    images = [Image.open(p).convert("RGB") for p in paths]
    max_h = max(im.height for im in images)
    resized = []
    for im in images:
        scale = max_h / im.height
        resized.append(im.resize((int(im.width * scale), max_h)))

    font_size = max(28, max_h // 18)
    font = _load_font(font_size)
    label_h = font_size + 36

    total_w = sum(im.width for im in resized)
    canvas = Image.new("RGB", (total_w, max_h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for im, label in zip(resized, labels):
        canvas.paste(im, (x, label_h))
        draw.rectangle((x, 0, x + im.width, label_h), fill=(30, 80, 120))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_h = bbox[3] - bbox[1]
        text_y = (label_h - text_h) // 2
        draw.text((x + 18, text_y), label, fill=(255, 255, 255), font=font)
        x += im.width
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    comp_dir = run_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)

    sam3_jsons = sorted((run_dir / "sam3").glob("*_sam3_result.json"))
    rows = []
    for sam3_json in sam3_jsons:
        stem = sam3_json.name.replace("_sam3_result.json", "")
        gs2_json = run_dir / "grounded_sam2" / f"{stem}_gs2_result.json"
        if not gs2_json.exists():
            continue
        sam3 = load_json(sam3_json)
        gs2 = load_json(gs2_json)
        rows.append(
            {
                "image": Path(sam3["image"]).name,
                "prompt": sam3["prompt"],
                "sam3_detections": len(sam3.get("detections", [])),
                "gs2_detections": len(gs2.get("detections", [])),
                "sam3_runtime_sec": f"{sam3.get('runtime_sec', 0):.3f}",
                "gs2_runtime_sec": f"{gs2.get('runtime_sec', 0):.3f}",
            }
        )
        prompt_str = sam3.get("prompt", "")
        make_side_by_side(
            [sam3["annotated_path"], gs2["annotated_path"]],
            [
                f'SAM3 — "{prompt_str}" — {rows[-1]["sam3_detections"]} detections',
                f'Grounded SAM2 — "{prompt_str}" — {rows[-1]["gs2_detections"]} detections',
            ],
            comp_dir / f"{stem}_side_by_side.jpg",
        )

    with open(comp_dir / "summary.csv", "w", newline="") as f:
        fieldnames = [
            "image",
            "prompt",
            "sam3_detections",
            "gs2_detections",
            "sam3_runtime_sec",
            "gs2_runtime_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    notes = comp_dir / "notes.md"
    if not notes.exists():
        notes.write_text(
            "# Comparison Notes\n\n- Fill qualitative notes here: mask quality, object separation, failure modes.\n"
        )
    print(f"Wrote {comp_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
