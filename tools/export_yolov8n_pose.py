"""Export Ultralytics YOLOv8n-Pose and write an independent numeric reference.

This script does not import VIE. Weights and the ONNX file stay outside the repository.
Pins are checked, not installed: ultralytics 8.4.0, torch 2.14.1, onnx 1.23.2.
"""

import hashlib
import os

import numpy as np
import onnx
import torch
from ultralytics import YOLO, __version__ as ultralytics_version

assert ultralytics_version == "8.4.0", ultralytics_version
assert torch.__version__.split("+")[0] == "2.14.1", torch.__version__
assert onnx.__version__ == "1.23.2", onnx.__version__

ROOT = "/tmp/vie_phase72"
PT = os.path.join(ROOT, "yolov8n-pose.pt")
os.makedirs(ROOT, exist_ok=True)

model = YOLO(PT)
head = model.model.model[-1]
print("nc", int(model.model.nc))
print("kpt_shape", tuple(int(value) for value in head.kpt_shape))
print("overrides", getattr(model, "overrides", None))

exported = model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
    simplify=False,
    opset=17,
    nms=False,
    device="cpu",
)
onnx_path = os.path.abspath(exported)
if os.path.dirname(onnx_path) != ROOT:
    target = os.path.join(ROOT, os.path.basename(onnx_path))
    os.replace(onnx_path, target)
    onnx_path = target
raw = open(onnx_path, "rb").read()
print("onnx", onnx_path)
print("onnx_bytes", len(raw))
print("onnx_sha256", hashlib.sha256(raw).hexdigest())
pt_raw = open(PT, "rb").read()
print("pt_bytes", len(pt_raw))
print("pt_sha256", hashlib.sha256(pt_raw).hexdigest())
print("torch", torch.__version__)
print("onnx_version", onnx.__version__)
print("ultralytics", ultralytics_version)

graph = onnx.load(onnx_path).graph
print("graph_input", [(item.name, [dim.dim_value for dim in item.type.tensor_type.shape.dim]) for item in graph.input])
print("graph_output", [(item.name, [dim.dim_value for dim in item.type.tensor_type.shape.dim], item.type.tensor_type.elem_type) for item in graph.output])
ops = {}
for node in graph.node:
    ops[node.op_type] = ops.get(node.op_type, 0) + 1
print("op_counts", dict(sorted(ops.items())))
print("has_nms", "NonMaxSuppression" in ops)
print("tail")
for node in graph.node[-25:]:
    print(node.op_type, list(node.input), list(node.output))

side = 640
count = side * side * 3
pattern = np.array([((index * 17 + 3) % 251) for index in range(count)], dtype=np.uint8)
pattern.tofile(os.path.join(ROOT, "input_u8.bin"))
open(os.path.join(ROOT, "image_side.txt"), "w").write(str(side) + "\n")
image = pattern.reshape(side, side, 3).astype(np.float32) / 255.0
tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous()
net = model.model
net.eval()
net.export = True
with torch.no_grad():
    prediction = net(tensor)
if isinstance(prediction, (tuple, list)):
    prediction = prediction[0]
array = prediction.detach().cpu().numpy().astype(np.float32)
array.tofile(os.path.join(ROOT, "raw_output.f32"))
np.savetxt(os.path.join(ROOT, "raw_shape.txt"), np.array(array.shape), fmt="%d")
print("output_shape", array.shape)

channels, anchors = int(array.shape[1]), int(array.shape[2])
nc = int(model.model.nc)
kpt_count, kpt_dim = (int(value) for value in head.kpt_shape)
assert channels == 4 + nc + kpt_count * kpt_dim, (channels, nc, kpt_count, kpt_dim)
assert kpt_dim == 3
boxes = array[0, 0:4, :]
scores = array[0, 4 : 4 + nc, :]
keypoints = array[0, 4 + nc :, :].reshape(kpt_count, kpt_dim, anchors)
best = scores.argmax(axis=0)
best_score = scores[best, np.arange(anchors)]
confidence = 0.0
pre_nms = 300
max_detections = 20
iou_threshold = 0.45


def xyxy(cx, cy, width, height):
    return cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5


def iou(left, right):
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    union = area_left + area_right - intersection
    if union <= 0:
        return 0.0
    return intersection / union


eligible = []
for index in range(anchors):
    score = float(best_score[index])
    if score < confidence:
        continue
    width = float(boxes[2, index])
    height = float(boxes[3, index])
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
        continue
    class_id = int(best[index])
    eligible.append((score, index, class_id))
eligible.sort(key=lambda item: (-item[0], item[1], item[2]))
eligible = eligible[:pre_nms]
mapped = []
for score, index, class_id in eligible:
    cx = float(boxes[0, index])
    cy = float(boxes[1, index])
    width = float(boxes[2, index])
    height = float(boxes[3, index])
    x1, y1, x2, y2 = xyxy(cx, cy, width, height)
    x1 = min(max(x1, 0.0), float(side))
    y1 = min(max(y1, 0.0), float(side))
    x2 = min(max(x2, 0.0), float(side))
    y2 = min(max(y2, 0.0), float(side))
    if not (x2 > x1 and y2 > y1):
        continue
    points = []
    for keypoint in range(kpt_count):
        px = float(keypoints[keypoint, 0, index])
        py = float(keypoints[keypoint, 1, index])
        confidence_point = float(keypoints[keypoint, 2, index])
        inside = px >= -1e-3 and px <= side + 1e-3 and py >= -1e-3 and py <= side + 1e-3
        points.append((px, py, confidence_point, inside))
    mapped.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1, class_id, score, index, (x1, y1, x2, y2), points))

kept = []
suppressed = [False] * len(mapped)
for left in range(len(mapped)):
    if suppressed[left]:
        continue
    kept.append(mapped[left])
    if len(kept) == max_detections:
        break
    for right in range(left + 1, len(mapped)):
        if suppressed[right]:
            continue
        if mapped[right][4] != mapped[left][4]:
            continue
        if iou(mapped[left][7], mapped[right][7]) >= iou_threshold:
            suppressed[right] = True

path = os.path.join(ROOT, "reference_poses.txt")
with open(path, "w") as handle:
    handle.write(
        "confidence 0 iou 0.45 pre_nms 300 max_detections 20 "
        "keypoints %d layout xy_confidence score_is_sigmoid_not_coco_visibility\n" % kpt_count
    )
    handle.write("%d %d\n" % (len(kept), kpt_count))
    for item in kept:
        handle.write("%s %s %s %s %d %.8g %d\n" % item[:7])
        for px, py, point_confidence, inside in item[8]:
            handle.write("%.8g %.8g %.8g %d\n" % (px, py, point_confidence, 1 if inside else 0))
print("kept", len(kept))
