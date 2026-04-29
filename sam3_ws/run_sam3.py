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

DATA_DIR = Path(os.environ.get("SAM3_DATA_DIR", "/home/simt-wj/03_Exp/ws_detection/data"))
OUT_DIR = Path(os.environ.get("SAM3_OUTPUT_DIR", "/home/simt-wj/03_Exp/ws_detection/data/annotated"))

text_prompts = ["ports"]


def nobj(x):
    return 0 if x is None else int(x.shape[0])


def load_model():
    if ck := os.environ.get("SAM3_CHECKPOINT"):
        if os.path.isfile(ck):
            return build_sam3_image_model(load_from_HF=False, checkpoint_path=ck)
    root = Path(__file__).resolve().parent
    hub = root / "huggingface_cache/hub/models--facebook--sam3/snapshots"
    if hub.is_dir():
        pts = sorted(hub.glob("*/sam3.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pts:
            return build_sam3_image_model(load_from_HF=False, checkpoint_path=str(pts[0]))
    return build_sam3_image_model()


def save_overlay(image, masks, boxes, scores, path, colors, alpha=0.5, labels=None):
    img = np.array(image.convert("RGB"))
    h, w = img.shape[:2]
    overlay = img.copy()
    masks = masks.cpu().float().numpy() if isinstance(masks, torch.Tensor) else masks
    boxes = boxes.cpu().float().numpy() if isinstance(boxes, torch.Tensor) else boxes
    scores = scores.cpu().float().numpy() if isinstance(scores, torch.Tensor) else scores
    if masks.ndim == 2:
        masks = masks[None, ...]
    if boxes.ndim == 1:
        boxes = boxes[None, ...]
    n = len(scores) if scores is not None else len(masks)

    for i in range(n):
        c = (colors[i % len(colors)] * 255).astype(np.uint8)
        m = masks[i].squeeze()
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        mb = m > 0.5
        for ch in range(3):
            overlay[..., ch][mb] = (alpha * c[ch] + (1 - alpha) * overlay[..., ch][mb]).astype(np.uint8)

    for i in range(n):
        x1, y1, x2, y2 = (int(x) for x in boxes[i])
        c = tuple(int(x * 255) for x in colors[i % len(colors)])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), c, 2)
        sc = scores[i] if scores is not None else None
        if labels and i < len(labels):
            text = f"{labels[i]}: {sc:.2f}" if sc is not None else labels[i]
        else:
            text = f"{sc:.2f}" if sc is not None else str(i)
        cv2.putText(overlay, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

    Image.fromarray(overlay).save(path)


def run_one(processor, image_path: Path, out_path: Path, colors):
    image = Image.open(image_path)
    t0 = time.perf_counter()
    state = processor.set_image(image)
    print(f"set_image: {(time.perf_counter() - t0) * 1000:.1f} ms")

    all_results = []
    t1 = time.perf_counter()
    for prompt in text_prompts:
        ps = state.copy()
        out = processor.set_text_prompt(prompt=prompt, state=ps)
        all_results.append(
            {"prompt": prompt, "masks": out["masks"], "boxes": out["boxes"], "scores": out["scores"]}
        )
        print(f"  prompt '{prompt}': {nobj(out['masks'])} mask(s)")
    print(f"text_prompt inference: {(time.perf_counter() - t1) * 1000:.1f} ms")
    print(f"inference total (set_image + text_prompts): {(time.perf_counter() - t0) * 1000:.1f} ms")

    if not all_results:
        masks = boxes = scores = None
    else:

        def cat(key):
            parts = [r[key] for r in all_results if r[key] is not None and len(r[key]) > 0]
            return torch.cat(parts, dim=0) if parts else None

        masks, boxes, scores = cat("masks"), cat("boxes"), cat("scores")

    print(f"masks (total): {nobj(masks)}")

    if nobj(masks) == 0:
        print("No detections, skip save.")
        return

    labels = []
    for r in all_results:
        labels.extend([r["prompt"]] * nobj(r["masks"]))

    t2 = time.perf_counter()
    save_overlay(image, masks, boxes, scores, str(out_path), colors, labels=labels)
    print(f"save overlay: {(time.perf_counter() - t2) * 1000:.1f} ms")
    print("saved:", out_path)


def main():
    try:
        model = load_model()

    except (HfHubHTTPError, LocalEntryNotFoundError) as e:
        print("Weights: set SAM3_CHECKPOINT, HF_TOKEN, or cache under huggingface_cache/...", e, file=sys.stderr)
        sys.exit(1)

    processor = Sam3Processor(model)
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    rng = np.random.default_rng(42)
    colors = rng.random((128, 3))

    if p := os.environ.get("SAM3_IMAGE"):
        paths = [Path(p).resolve()] if Path(p).is_file() else []
    else:
        paths = sorted(DATA_DIR.glob("*.ppm"))

    if not paths:
        print(f"No .ppm files in {DATA_DIR} (set SAM3_IMAGE for a single file)", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"data: {DATA_DIR}")
    print(f"output: {OUT_DIR}")
    print(f"prompts: {text_prompts}")
    print(f"files: {len(paths)}")

    for image_path in paths:
        print("---", image_path.name, "---")
        out_path = OUT_DIR / f"{image_path.stem}_result.jpg"
        with autocast_ctx:
            run_one(processor, image_path, out_path, colors)


if __name__ == "__main__":
    main()
