"""Export Ultralytics YOLOv8n-OBB and write an independent numeric reference.

This script does not import VIE. Weights stay outside the repository.
Pins are checked, not installed: ultralytics 8.4.0, torch 2.14.1, onnx 1.23.2.
"""

import hashlib
import math
import os

import numpy as np
import torch
from ultralytics import YOLO, __version__ as ultralytics_version

assert ultralytics_version == "8.4.0", ultralytics_version
assert torch.__version__.split("+")[0] == "2.14.1", torch.__version__

ROOT = "/tmp/vie_phase71"
PT = os.path.join(ROOT, "yolov8n-obb.pt")
os.makedirs(ROOT, exist_ok=True)

model = YOLO(PT)
exported = model.export(
    format="onnx",
    imgsz=1024,
    batch=1,
    dynamic=False,
    simplify=False,
    opset=17,
    nms=False,
    device="cpu",
)
onnx_path = os.path.abspath(exported)
raw = open(onnx_path, "rb").read()
print("onnx", onnx_path)
print("onnx_bytes", len(raw))
print("onnx_sha256", hashlib.sha256(raw).hexdigest())
pt_raw = open(PT, "rb").read()
print("pt_bytes", len(pt_raw))
print("pt_sha256", hashlib.sha256(pt_raw).hexdigest())

side = 1024
count = side * side * 3
pattern = np.array([((index * 17 + 3) % 251) for index in range(count)], dtype=np.uint8)
pattern.tofile(os.path.join(ROOT, "input_u8.bin"))
image = pattern.reshape(side, side, 3).astype(np.float32) / 255.0
tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous()
net = model.model
net.eval()
with torch.no_grad():
    prediction = net(tensor)
if isinstance(prediction, (tuple, list)):
    prediction = prediction[0]
array = prediction.detach().cpu().numpy().astype(np.float32)
array.tofile(os.path.join(ROOT, "raw_output.f32"))
np.savetxt(os.path.join(ROOT, "raw_shape.txt"), np.array(array.shape), fmt="%d")
print("output_shape", array.shape)

# Independent decode: best class, no second sigmoid, polygon IoU NMS.
# This is not ultralytics.utils.nms, which uses probabilistic IoU.
channels, anchors = array.shape[1], array.shape[2]
classes = channels - 5
boxes = array[0, :4, :]
scores = array[0, 4 : 4 + classes, :]
angles = array[0, -1, :]
best = scores.argmax(axis=0)
best_score = scores[best, np.arange(anchors)]
confidence = 0.0
pre_nms = 300
max_detections = 20
iou_threshold = 0.45
eligible = []
for index in range(anchors):
    score = float(best_score[index])
    if score < confidence:
        continue
    width = float(boxes[2, index])
    height = float(boxes[3, index])
    angle = float(angles[index])
    if width <= 0 or height <= 0 or angle < -math.pi / 4 - 1e-3 or angle > 3 * math.pi / 4 + 1e-3:
        raise SystemExit(f"kept anchor {index} has an illegal size or angle")
    eligible.append(index)
eligible.sort(key=lambda index: (-float(best_score[index]), index))
selected = eligible[:pre_nms]


def corners(cx, cy, w, h, angle):
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    vx = (w * 0.5 * cosine, w * 0.5 * sine)
    vy = (-h * 0.5 * sine, h * 0.5 * cosine)
    center = (cx, cy)
    return [
        (center[0] + vx[0] + vy[0], center[1] + vx[1] + vy[1]),
        (center[0] + vx[0] - vy[0], center[1] + vx[1] - vy[1]),
        (center[0] - vx[0] - vy[0], center[1] - vx[1] - vy[1]),
        (center[0] - vx[0] + vy[0], center[1] - vx[1] + vy[1]),
    ]


def signed_area(points):
    total = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        total += point[0] * nxt[1] - point[1] * nxt[0]
    return total * 0.5


def clip(subject, clipper):
    output = subject
    for edge in range(len(clipper)):
        if not output:
            break
        a = clipper[edge]
        b = clipper[(edge + 1) % len(clipper)]
        incoming = output
        output = []
        for index, start in enumerate(incoming):
            end = incoming[(index + 1) % len(incoming)]
            def inside(point):
                return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) >= -1e-9
            def hit(s, e):
                se = (e[0] - s[0], e[1] - s[1])
                ab = (b[0] - a[0], b[1] - a[1])
                denom = se[0] * ab[1] - se[1] * ab[0]
                if abs(denom) < 1e-18:
                    return e
                t = ((a[0] - s[0]) * ab[1] - (a[1] - s[1]) * ab[0]) / denom
                t = min(1.0, max(0.0, t))
                return (s[0] + se[0] * t, s[1] + se[1] * t)
            s_in, e_in = inside(start), inside(end)
            if s_in and e_in:
                output.append(end)
            elif s_in and not e_in:
                output.append(hit(start, end))
            elif not s_in and e_in:
                output.append(hit(start, end))
                output.append(end)
    return output


def iou(left, right):
    lc = corners(*left)
    rc = corners(*right)
    if signed_area(lc) < 0:
        lc.reverse()
    if signed_area(rc) < 0:
        rc.reverse()
    inter = abs(signed_area(clip(lc, rc)))
    left_area = abs(signed_area(lc))
    right_area = abs(signed_area(rc))
    if inter > left_area:
        inter = left_area
    if inter > right_area:
        inter = right_area
    union = left_area + right_area - inter
    return 0.0 if union <= 0 else inter / union


def leaves_image(box):
    for x, y in corners(*box):
        if x < -1e-3 or y < -1e-3 or x > side + 1e-3 or y > side + 1e-3:
            return True
    return False


kept = []
for index in selected:
    candidate = (
        float(boxes[0, index]),
        float(boxes[1, index]),
        float(boxes[2, index]),
        float(boxes[3, index]),
        float(angles[index]),
        int(best[index]),
        float(best_score[index]),
        int(index),
    )
    if leaves_image(candidate[:5]):
        continue
    suppress = False
    for previous in kept:
        if previous[5] != candidate[5]:
            continue
        if iou(previous[:5], candidate[:5]) >= iou_threshold:
            suppress = True
            break
    if not suppress:
        kept.append(candidate)
    if len(kept) == max_detections:
        break

with open(os.path.join(ROOT, "reference_boxes.txt"), "w", encoding="utf-8") as handle:
    handle.write(
        f"classes {classes} anchors {anchors} confidence {confidence} iou {iou_threshold} "
        f"pre_nms {pre_nms} max_detections {max_detections}\n"
    )
    for item in kept:
        handle.write(" ".join(str(part) for part in item) + "\n")
print("kept", len(kept))
print("max_class_score", float(array[0, 4:-1, :].max()))

import onnx

graph = onnx.load(onnx_path, load_external_data=False).graph
print("input", graph.input[0].name)
print("output", graph.output[0].name)
counts = {}
for node in graph.node:
    counts[node.op_type] = counts.get(node.op_type, 0) + 1
print("ops", " ".join(f"{name}:{counts[name]}" for name in sorted(counts)))
print("final", graph.node[-1].op_type, list(graph.node[-1].input), list(graph.node[-1].output))
