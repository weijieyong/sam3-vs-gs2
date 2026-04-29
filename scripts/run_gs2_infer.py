#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv
import torch
from PIL import Image
from torchvision.ops import box_convert

sys.path.insert(0, str(Path(os.getcwd()) / "source"))
sys.path.insert(0, str(Path(os.getcwd()) / "source" / "grounding_dino"))
from grounding_dino.groundingdino.util.inference import load_image, load_model, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def write_mask_png(image_shape, masks, path):
    h, w = image_shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    if masks is not None:
        for i, m in enumerate(masks, start=1):
            combined[m.astype(bool)] = min(i, 255)
    Image.fromarray(combined).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--box-threshold", type=float, default=0.35)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2-model-config", default="source/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--gdino-config", default="source/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--gdino-checkpoint", default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    args = ap.parse_args()

    image_path = Path(args.image).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    text = args.prompt.lower().strip()
    if not text.endswith("."):
        text += "."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    t0 = time.perf_counter()
    sam2_model = build_sam2(args.sam2_model_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    grounding_model = load_model(args.gdino_config, args.gdino_checkpoint, device=device)

    image_source, image = load_image(str(image_path))
    sam2_predictor.set_image(image_source)
    boxes, confidences, class_names = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device,
    )

    h, w, _ = image_source.shape
    input_boxes = np.empty((0, 4), dtype=np.float32)
    masks = np.empty((0, h, w), dtype=bool)
    scores = []
    if len(boxes) > 0:
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            masks, scores, _ = sam2_predictor.predict(point_coords=None, point_labels=None, box=input_boxes, multimask_output=False)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
    runtime = time.perf_counter() - t0

    confidences_list = confidences.cpu().numpy().tolist() if hasattr(confidences, "cpu") else list(confidences)
    class_ids = np.array(list(range(len(class_names))))
    labels = [f"{name} {score:.2f}" for name, score in zip(class_names, confidences_list)]

    annotated = out_dir / f"{stem}_gs2_annotated.jpg"
    mask_path = out_dir / f"{stem}_gs2_mask.png"
    result_path = out_dir / f"{stem}_gs2_result.json"

    img = cv2.imread(str(image_path))
    if len(input_boxes) > 0:
        detections_sv = sv.Detections(xyxy=input_boxes, mask=masks.astype(bool), class_id=class_ids)
        annotated_frame = sv.BoxAnnotator().annotate(scene=img.copy(), detections=detections_sv)
        annotated_frame = sv.LabelAnnotator().annotate(scene=annotated_frame, detections=detections_sv, labels=labels)
        annotated_frame = sv.MaskAnnotator().annotate(scene=annotated_frame, detections=detections_sv)
        cv2.imwrite(str(annotated), annotated_frame)
    else:
        cv2.imwrite(str(annotated), img)
    write_mask_png(image_source.shape, masks, mask_path)

    detections = []
    for i, name in enumerate(class_names):
        detections.append({
            "label": name,
            "score": float(confidences_list[i]),
            "bbox": [float(x) for x in input_boxes[i].tolist()],
            "segmentation": single_mask_to_rle(masks[i].astype(np.uint8)) if len(masks) > i else None,
        })
    payload = {
        "model": "grounded_sam2",
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
