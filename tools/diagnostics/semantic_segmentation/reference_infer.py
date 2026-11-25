#!/usr/bin/env python3
"""Phase 9.4 numeric reference for the DeepLabV3 semantic-segmentation task.

Runs the ORIGINAL torchvision model on CPU, then compares three artifacts
produced by vie_semantic_verify against this reference:

  * <prefix>_input.f32   VIE preprocessed NCHW float32 tensor (shape 1,3,520,520)
  * <prefix>_logits.f32  VIE backend raw logits tensor (shape 1,21,520,520)
  * <prefix>_classids.i32 VIE decoder per-pixel class map in original coordinates

The reference transform is the same torchvision DeepLabV3 eval contract the
manifest records as User provenance: bilinear stretch to 520x520, RGB channel
order, ImageNet mean/std normalization. The reference decode step interpolates
the logits back to the original size with bilinear + align_corners=False and
picks class = argmax channel. Neither step fabricates values; both are read
straight from the persisted artifacts so a mismatch fails loudly.

Nothing in this script mutates the model or the ONNX file. It only reports
numeric deltas (MaxAbs, RMSE, correlation, PixelAccuracy, MeanIoU) for the
table in the Phase 9.4 delivery doc. CUDA vs CPU comparison uses the exact
same reference numbers as a common denominator.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


VOC_CATEGORIES = [
    "__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
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
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
    return w, h, arr


def reference_preprocess(rgb_hwc: np.ndarray, size: int) -> np.ndarray:
    """Torchvision DeepLabV3 eval contract, pinned to VIE's exact math.

    VIE preprocess uses half-pixel-centered bilinear sampling with
    align_corners=False (source_coordinate = (dst+0.5)*src/dst - 0.5). PIL
    ``resize`` with antialias=False runs a slightly different kernel on
    non-integer downscales, so we reproduce the VIE formula with
    ``F.interpolate(..., mode='bilinear', align_corners=False)``. Normalization
    stays the torchvision ToTensor + Normalize pair: RGB -> /255 -> minus
    ImageNet mean -> divided by ImageNet std.
    """
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(rgb_hwc.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    normalized = (resized - mean) / std
    return normalized[0].numpy()  # 3, H, W


def reference_logits(input_nchw: np.ndarray) -> np.ndarray:
    """Run the torchvision DeepLabV3 (COCO_WITH_VOC_LABELS_V1) on the CPU."""
    import torch
    from torchvision.models.segmentation import (
        deeplabv3_mobilenet_v3_large,
        DeepLabV3_MobileNet_V3_Large_Weights,
    )

    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights, num_classes=21).eval()
    with torch.no_grad():
        out = model(torch.from_numpy(input_nchw[None, ...]))["out"]
    return out.numpy()


def reference_class_map(logits_nhwc_cls: np.ndarray, original_hw: Tuple[int, int]) -> np.ndarray:
    """F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False).argmax(1)."""
    import torch
    import torch.nn.functional as F

    n, c, h, w = logits_nhwc_cls.shape
    H, W = original_hw
    up = F.interpolate(torch.from_numpy(logits_nhwc_cls), size=(H, W), mode="bilinear",
                       align_corners=False)
    return up.argmax(dim=1)[0].numpy().astype(np.int32)  # H, W


def load_f32(path: str, shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"{path}: expected {expected} float32, got {arr.size}")
    return arr.reshape(shape)


def load_i32(path: str, shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.int32)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"{path}: expected {expected} int32, got {arr.size}")
    return arr.reshape(shape)


def f32_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.abs(np.asarray(b, dtype=np.float64)), 1e-12)
    return {
        "count": int(diff.size),
        "nan_a": int(np.isnan(a).sum()),
        "nan_b": int(np.isnan(b).sum()),
        "inf_a": int(np.isinf(a).sum()),
        "inf_b": int(np.isinf(b).sum()),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_rel": float(np.max(np.abs(diff) / denom)),
        "cosine": float(np.dot(a.reshape(-1), b.reshape(-1)) /
                        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
    }


def class_map_stats(vie: np.ndarray, ref: np.ndarray, num_classes: int) -> Dict[str, object]:
    vie = vie.reshape(-1).astype(np.int64)
    ref = ref.reshape(-1).astype(np.int64)
    disagreement = int(np.sum(vie != ref))
    pixel_acc = 1.0 - disagreement / max(vie.size, 1)
    ious: List[Optional[float]] = []
    for cls in range(num_classes):
        a = (vie == cls)
        b = (ref == cls)
        inter = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        ious.append(None if union == 0 else inter / union)
    valid = [x for x in ious if x is not None]
    mean_iou = float(np.mean(valid)) if valid else 0.0
    return {
        "pixel_accuracy": pixel_acc,
        "disagreement_pixels": disagreement,
        "total_pixels": int(vie.size),
        "disagreement_ratio": disagreement / max(vie.size, 1),
        "mean_iou": mean_iou,
        "per_class_iou": [{"class": cls, "name": VOC_CATEGORIES[cls] if cls < len(VOC_CATEGORIES) else str(cls), "iou": iou} for cls, iou in enumerate(ious)],
        "out_of_range_ids": int(np.sum((vie < 0) | (vie >= num_classes))),
    }


def compare_one(name: str, prefix: str, size: int, orig_hw: Tuple[int, int],
                ref_input: np.ndarray, ref_logits: np.ndarray, ref_class: np.ndarray,
                out: Dict[str, object]) -> None:
    input_path = f"{prefix}_input.f32"
    logits_path = f"{prefix}_logits.f32"
    class_path = f"{prefix}_classids.i32"
    if not (os.path.exists(input_path) and os.path.exists(logits_path) and os.path.exists(class_path)):
        out[name] = {"status": "missing_artifacts", "prefix": prefix}
        return
    vie_input = load_f32(input_path, (1, 3, size, size))
    vie_logits = load_f32(logits_path, (1, 21, size, size))
    vie_class = load_i32(class_path, orig_hw)
    out[name] = {
        "status": "ok",
        "preprocess": f32_stats(vie_input, ref_input),
        "logits": f32_stats(vie_logits, ref_logits),
        "class_map": class_map_stats(vie_class, ref_class, num_classes=21),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to bus.ppm (P6)")
    ap.add_argument("--size", type=int, default=520)
    ap.add_argument("--cpu-prefix", default="/tmp/sem_out/cpu")
    ap.add_argument("--gpu-prefix", default="/tmp/sem_out/gpu")
    ap.add_argument("--out", default="/tmp/sem_out/reference_report.json")
    args = ap.parse_args()

    w, h, rgb = read_ppm(args.image)
    ref_input = reference_preprocess(rgb, args.size)
    ref_logits = reference_logits(ref_input)
    ref_class = reference_class_map(ref_logits, (h, w))

    report: Dict[str, object] = {
        "image": {"path": args.image, "width": w, "height": h},
        "model": {
            "name": "DeepLabV3-MobileNetV3-Large",
            "weights": "COCO_WITH_VOC_LABELS_V1 (torchvision default)",
            "num_classes": 21,
            "input_size": args.size,
            "logits_shape": list(ref_logits.shape),
            "class_map_shape": list(ref_class.shape),
        },
        "reference_self_check": {
            "input_dtype": "float32",
            "input_range": [float(ref_input.min()), float(ref_input.max())],
            "input_nan": int(np.isnan(ref_input).sum()),
            "logits_nan": int(np.isnan(ref_logits).sum()),
            "class_id_range": [int(ref_class.min()), int(ref_class.max())],
            "top_classes_present": sorted(set(int(x) for x in np.unique(ref_class))),
        },
    }
    compare_one("cpu_vie", args.cpu_prefix, args.size, (h, w), ref_input, ref_logits, ref_class, report)
    compare_one("cuda_vie", args.gpu_prefix, args.size, (h, w), ref_input, ref_logits, ref_class, report)

    # Cross-device: CPU vs CUDA within VIE (both must be structurally identical).
    if os.path.exists(args.cpu_prefix + "_logits.f32") and os.path.exists(args.gpu_prefix + "_logits.f32"):
        cpu_logits = load_f32(args.cpu_prefix + "_logits.f32", (1, 21, args.size, args.size))
        gpu_logits = load_f32(args.gpu_prefix + "_logits.f32", (1, 21, args.size, args.size))
        cpu_cls = load_i32(args.cpu_prefix + "_classids.i32", (h, w))
        gpu_cls = load_i32(args.gpu_prefix + "_classids.i32", (h, w))
        report["cpu_vs_cuda"] = {
            "logits": f32_stats(cpu_logits, gpu_logits),
            "class_map": class_map_stats(cpu_cls, gpu_cls, num_classes=21),
        }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
