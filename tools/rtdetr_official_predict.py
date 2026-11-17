#!/usr/bin/env python3
"""Official Ultralytics RT-DETR predict for one image.

Writes detections in original-image pixels and the raw eval tensor. This is a
test aid. It does not import VIE.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import RTDETR


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: rtdetr_official_predict.py <image> <weight.pt> <out.txt>")
    image_path, weight_path, out_path = sys.argv[1:]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cv2 could not read {image_path}")
    model = RTDETR(weight_path)
    results = model.predict(source=image, conf=0.25, max_det=300, verbose=False)
    result = results[0]
    lines = [f"image {image.shape[1]} {image.shape[0]}"]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        lines.append("count 0")
    else:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        lines.append(f"count {len(classes)}")
        for index in range(len(classes)):
            lines.append(
                f"det {classes[index]} {conf[index]:.8f} "
                f"{xyxy[index, 0]:.6f} {xyxy[index, 1]:.6f} "
                f"{xyxy[index, 2]:.6f} {xyxy[index, 3]:.6f}"
            )

    batch = model.predictor.preprocess([image])
    preprocessed = batch["img"] if isinstance(batch, dict) else batch
    with torch.no_grad():
        raw = model.model(preprocessed)
    pre_numpy = preprocessed.detach().float().cpu().numpy()
    pre_path = str(Path(out_path).with_suffix(".pre.f32"))
    pre_numpy.astype(np.float32).tofile(pre_path)
    lines.append("pre_shape " + " ".join(str(dim) for dim in pre_numpy.shape))
    lines.append(f"pre_file {pre_path}")
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    raw = raw.detach().float().cpu().numpy()
    raw_path = str(Path(out_path).with_suffix(".f32"))
    raw.astype(np.float32).tofile(raw_path)
    lines.append("raw_shape " + " ".join(str(dim) for dim in raw.shape))
    lines.append(f"raw_file {raw_path}")
    Path(out_path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
