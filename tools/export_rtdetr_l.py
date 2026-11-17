"""Export Ultralytics RT-DETR-L to a static ONNX file.

This script does not import or modify VIE. It records the exact export used for
testdata/onnx/rtdetr-l.onnx.
"""

from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")
path = model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
    simplify=False,
    opset=17,
    nms=False,
    device="cpu",
)
print(path)
