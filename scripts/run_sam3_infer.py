#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def nobj(x):
    return 0 if x is None else int(x.shape[0])


def load_model():
    if ck := os.environ.get("SAM3_CHECKPOINT"):
        if os.path.isfile(ck):
            return build_sam3_image_model(load_from_HF=False, checkpoint_path=ck)
    root = (
        Path(__file__).resolve().parents[2]
    )  # overridden if copied; fallback below is cwd
    cwd = Path.cwd()
    for base in [cwd, root]:
        hub = base / "huggingface_cache/hub/models--facebook--sam3/snapshots"
        if hub.is_dir():
            pts = sorted(
                hub.glob("*/sam3.pt"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if pts:
                return build_sam3_image_model(
                    load_from_HF=False, checkpoint_path=str(pts[0])
                )
    return build_sam3_image_model()


def save_overlay(image, masks, boxes, scores, labels, path, alpha=0.5):
    img = np.array(image.convert("RGB"))
    h, w = img.shape[:2]
    overlay = img.copy()
    masks = masks.cpu().float().numpy() if isinstance(masks, torch.Tensor) else masks
    boxes = boxes.cpu().float().numpy() if isinstance(boxes, torch.Tensor) else boxes
    scores = (
        scores.cpu().float().numpy() if isinstance(scores, torch.Tensor) else scores
    )
    if masks.ndim == 2:
        masks = masks[None, ...]
    if boxes.ndim == 1:
        boxes = boxes[None, ...]
    rng = np.random.default_rng(42)
    colors = rng.random((max(len(masks), 1), 3))
    for i in range(len(masks)):
        c = (colors[i] * 255).astype(np.uint8)
        m = masks[i].squeeze()
        if m.shape != (h, w):
            m = cv2.resize(
                m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
            )
        mb = m > 0.5
        for ch in range(3):
            overlay[..., ch][mb] = (
                alpha * c[ch] + (1 - alpha) * overlay[..., ch][mb]
            ).astype(np.uint8)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (int(x) for x in boxes[i])
        c = tuple(int(x * 255) for x in colors[i % len(colors)])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), c, 2)
        label = labels[i] if i < len(labels) else "sam3"
        score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
        cv2.putText(
            overlay,
            f"{label}: {score:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            c,
            2,
            cv2.LINE_AA,
        )
    Image.fromarray(overlay).save(path)


def write_mask_png(image, masks, path):
    w, h = image.size
    if nobj(masks) == 0:
        Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(path)
        return
    arr = masks.cpu().float().numpy() if isinstance(masks, torch.Tensor) else masks
    if arr.ndim == 2:
        arr = arr[None, ...]
    combined = np.zeros((h, w), dtype=np.uint8)
    for i, m in enumerate(arr, start=1):
        m = m.squeeze()
        if m.shape != (h, w):
            m = cv2.resize(
                m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
            )
        combined[m > 0.5] = min(i, 255)
    Image.fromarray(combined).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument(
        "--prompt",
        required=True,
        help="Use semicolon or comma to separate multiple prompts",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score × presence score to keep a detection (default: 0.5)",
    )
    args = ap.parse_args()

    image_path = Path(args.image).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    # GS2/GroundingDINO convention often uses dot-separated prompts: "car. tire."
    # SAM3 expects one text concept per call, so split common prompt separators here.
    raw_prompt = args.prompt.replace(";", ",")
    if "," in raw_prompt:
        prompts = [
            p.strip().strip(".") for p in raw_prompt.split(",") if p.strip().strip(".")
        ]
    else:
        prompts = [p.strip() for p in raw_prompt.split(".") if p.strip()]

    try:
        model = load_model()
    except (HfHubHTTPError, LocalEntryNotFoundError) as e:
        print("SAM3 weights unavailable", e, file=sys.stderr)
        return 2

    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)
    image = Image.open(image_path).convert("RGB")
    t0 = time.perf_counter()
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
    ):
        state = processor.set_image(image)
        all_results = []
        for prompt in prompts:
            out = processor.set_text_prompt(prompt=prompt, state=state.copy())
            all_results.append(
                {
                    "prompt": prompt,
                    "masks": out["masks"],
                    "boxes": out["boxes"],
                    "scores": out["scores"],
                }
            )
    runtime = time.perf_counter() - t0

    def cat(key):
        parts = [r[key] for r in all_results if r[key] is not None and len(r[key]) > 0]
        return torch.cat(parts, dim=0) if parts else None

    masks, boxes, scores = cat("masks"), cat("boxes"), cat("scores")
    labels = []
    for r in all_results:
        labels.extend([r["prompt"]] * nobj(r["masks"]))

    annotated = out_dir / f"{stem}_sam3_annotated.jpg"
    mask_path = out_dir / f"{stem}_sam3_mask.png"
    result_path = out_dir / f"{stem}_sam3_result.json"

    detections = []
    if nobj(masks) > 0:
        save_overlay(image, masks, boxes, scores, labels, annotated)
        write_mask_png(image, masks, mask_path)
        b = boxes.cpu().float().numpy().tolist()
        s = scores.cpu().float().numpy().tolist()
        detections = [
            {
                "label": labels[i] if i < len(labels) else "",
                "score": float(s[i]),
                "bbox": [float(x) for x in b[i]],
            }
            for i in range(len(b))
        ]
    else:
        image.save(annotated)
        write_mask_png(image, None, mask_path)

    payload = {
        "model": "sam3",
        "image": str(image_path),
        "prompt": args.prompt,
        "runtime_sec": runtime,
        "detections": detections,
        "annotated_path": str(annotated),
        "mask_path": str(mask_path),
    }
    result_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
