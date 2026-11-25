#!/usr/bin/env python3
"""Phase 9.4 semantic-segmentation class-map visualizer.

Reads the raw int32 class map that ``vie_semantic_verify`` dumps
(``<prefix>_classids.i32``, row-major, width*height elements, values in
``[0, num_classes)``) plus the source image (P6 PPM or any PIL-decodable
file), and writes an overlay PNG where each class gets a fixed palette
colour. Two outputs are produced by default:

  * ``<out>.mask.png``   pure colour-coded class map
  * ``<out>.overlay.png`` 60/40 blend of the original image and the mask

The palette follows the COG/PASCAL VOC convention used by the DeepLabV3
COCO_WITH_VOC_LABELS_V1 checkpoint, matching the categories the model's
manifest records. Background (class 0) is drawn as a translucent dark
shade so the image still shows through in the overlay. This script never
runs inference and never edits the model; it is a viewer over VIE's own
artifacts.

Example:
    python3 visualize_result.py \
        --class-ids /tmp/sem_out/cpu_classids.i32 \
        --image testdata/images/bus.ppm \
        --width 810 --height 1080 \
        --out /tmp/sem_out/cpu
"""
import argparse
import os
import sys
from typing import List, Tuple

import numpy as np


# COG / PASCAL-VOC 21-class palette. Index 0 is __background__.
PALETTE: List[Tuple[int, int, int]] = [
    (0, 0, 0),           # background
    (128, 0, 0),         # aeroplane
    (0, 128, 0),         # bicycle
    (128, 128, 0),       # bird
    (0, 0, 128),         # boat
    (128, 0, 128),       # bottle
    (0, 128, 128),       # bus
    (128, 128, 128),     # car
    (64, 0, 0),          # chair
    (192, 0, 0),         # cow
    (64, 128, 0),        # diningtable
    (192, 128, 0),       # dog
    (64, 0, 128),        # horse
    (192, 0, 128),       # motorbike
    (64, 128, 128),      # person
    (192, 128, 128),     # pottedplant
    (0, 64, 0),          # sheep
    (128, 64, 0),        # sofa
    (0, 192, 0),         # train
    (128, 192, 0),       # tvmonitor
    (255, 255, 255),     # class 20 (unused in VOC, kept for safety)
]

VOC_NAMES: List[str] = [
    "__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "extra",
]


def read_ppm(path: str) -> Tuple[int, int, np.ndarray]:
    """Read a binary P6 PPM into an HWC uint8 RGB array."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P6":
            raise ValueError(f"{path}: expected P6 PPM, got {magic!r}")
        line = f.readline().strip()
        while line.startswith(b"#"):
            line = f.readline().strip()
        w, h = (int(x) for x in line.split())
        maxv = int(f.readline().strip())
        if maxv != 255:
            raise ValueError(f"{path}: only max 255 supported, got {maxv}")
        raw = f.read(w * h * 3)
        if len(raw) != w * h * 3:
            raise ValueError(f"{path}: short read")
    return w, h, np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))


def read_any_image(path: str) -> Tuple[int, int, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ppm":
        return read_ppm(path)
    from PIL import Image  # local import; only non-PPM path needs it
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return img.width, img.height, arr


def load_class_ids(path: str, width: int, height: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.int32)
    expected = width * height
    if arr.size != expected:
        raise ValueError(f"{path}: expected {expected} int32 pixels, got {arr.size}")
    return arr.reshape((height, width))


def colorize(class_ids: np.ndarray, num_classes: int) -> np.ndarray:
    palette = np.array(PALETTE[:num_classes], dtype=np.uint8)
    if palette.shape[0] < num_classes:
        # Extend deterministically with a dark ramp so nothing reads as unset.
        extra = np.zeros((num_classes - palette.shape[0], 3), dtype=np.uint8)
        for k in range(palette.shape[0], num_classes):
            extra[k - palette.shape[0]] = ((k * 37) & 0xFF, (k * 91) & 0xFF, (k * 53) & 0xFF)
        palette = np.concatenate([palette, extra], axis=0)
    clipped = np.clip(class_ids, 0, palette.shape[0] - 1)
    return palette[clipped]


def save_png(path: str, rgb: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def summarize(class_ids: np.ndarray, num_classes: int) -> str:
    counts = np.bincount(class_ids.reshape(-1).astype(np.int64), minlength=num_classes)
    total = int(counts.sum())
    lines = [f"class_id histogram ({total} pixels, num_classes={num_classes}):"]
    for k, cnt in enumerate(counts):
        if cnt == 0:
            continue
        name = VOC_NAMES[k] if k < len(VOC_NAMES) else str(k)
        lines.append(f"  [{k:2d}] {name:<14s} {cnt:>10d}  {cnt / total * 100.0:5.2f}%")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-ids", required=True, help="<prefix>_classids.i32 from vie_semantic_verify")
    ap.add_argument("--image", required=True, help="source image (P6 PPM or PIL-decodable)")
    ap.add_argument("--width", type=int, default=0, help="0 = infer from --image")
    ap.add_argument("--height", type=int, default=0, help="0 = infer from --image")
    ap.add_argument("--num-classes", type=int, default=21)
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="mask weight in overlay (0 = image only, 1 = mask only)")
    ap.add_argument("--out", default="/tmp/sem_out/visual",
                    help="output path stem; produces <stem>.mask.png and <stem>.overlay.png")
    ap.add_argument("--skip-overlay", action="store_true")
    args = ap.parse_args()

    iw, ih, rgb = read_any_image(args.image)
    w = args.width or iw
    h = args.height or ih
    if (w, h) != (iw, ih):
        print(f"warning: class map grid {w}x{h} differs from image size {iw}x{ih}",
              file=sys.stderr)
    class_ids = load_class_ids(args.class_ids, w, h)
    if int(class_ids.min()) < 0 or int(class_ids.max()) >= args.num_classes:
        raise ValueError(
            f"class ids out of range [0,{args.num_classes}): "
            f"min={int(class_ids.min())} max={int(class_ids.max())}"
        )

    mask = colorize(class_ids, args.num_classes)
    save_png(args.out + ".mask.png", mask)
    if not args.skip_overlay and (w, h) == (iw, ih):
        alpha = max(0.0, min(1.0, args.alpha))
        blended = (rgb.astype(np.float32) * (1.0 - alpha) +
                   mask.astype(np.float32) * alpha)
        save_png(args.out + ".overlay.png", np.clip(blended, 0, 255).astype(np.uint8))

    print(summarize(class_ids, args.num_classes))
    print(f"wrote {args.out}.mask.png" + ("" if args.skip_overlay else f" and {args.out}.overlay.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
