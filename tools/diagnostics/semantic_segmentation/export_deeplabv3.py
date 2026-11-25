#!/usr/bin/env python3
"""Reproducible DeepLabV3-MobileNetV3-Large ONNX export (torchvision weights).

Produces a single-4D-logit-output ONNX at opset 18 (fixed 1x3x520x520) using the
OFFICIAL torchvision pretrained weights, and records the structural facts the VIE
recognizer and manifest depend on. It never fabricates weights: if the pretrained
checkpoint cannot be fetched it aborts rather than substituting a random model.
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Wrapper(__import__("torch").nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="testdata/onnx/deeplabv3_mobilenet_v3_large_520_opset18.onnx")
    ap.add_argument("--size", type=int, default=520)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--info", default="testdata/onnx/deeplabv3_mobilenet_v3_large_520_opset18.info.json")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="export architecture only (random init). NEVER label this as pretrained.")
    args = ap.parse_args()

    import torch
    import torchvision
    from torchvision.models.segmentation import (
        deeplabv3_mobilenet_v3_large,
        DeepLabV3_MobileNet_V3_Large_Weights,
    )

    if args.no_pretrained:
        weights = None
        source = "architecture-only (random init, NOT pretrained)"
    else:
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        source = str(weights)
    model = deeplabv3_mobilenet_v3_large(weights=weights, num_classes=21)
    model.eval()
    categories = None
    if not args.no_pretrained:
        categories = weights.meta.get("categories")

    wrapper = Wrapper(model)
    wrapper.eval()
    dummy = torch.zeros(1, 3, args.size, args.size)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            args.out,
            opset_version=args.opset,
            input_names=["image"],
            output_names=["out"],
            dynamic_axes=None,
            dynamo=False,
            do_constant_folding=True,
        )

    import onnx
    from onnx import shape_inference

    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    m_inferred = shape_inference.infer_shapes(m, strict_mode=False)

    opset = None
    for op in m.opset_import:
        if op.domain in ("", "ai.onnx"):
            opset = op.version
    inputs = []
    for vi in m.graph.input:
        shape = []
        dt = None
        t = vi.type.tensor_type
        dt = onnx.TensorProto.DataType.Name(t.elem_type)
        for d in t.shape.dim:
            shape.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
        inputs.append({"name": vi.name, "shape": shape, "dtype": dt})
    outputs = []
    for vi in m.graph.output:
        shape = []
        t = vi.type.tensor_type
        dt = onnx.TensorProto.DataType.Name(t.elem_type)
        for d in t.shape.dim:
            shape.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
        outputs.append({"name": vi.name, "shape": shape, "dtype": dt})

    hist = collections.Counter(n.op_type for n in m.graph.node)
    # last node op type chain (topological order of graph.node)
    last_nodes = [n.op_type for n in m.graph.node[-6:]]

    info = {
        "model": "DeepLabV3-MobileNetV3-Large",
        "weights_source": source,
        "pretrained": not args.no_pretrained,
        "num_classes": 21,
        "categories": categories,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "onnx_version": onnx.__version__,
        "opset": opset,
        "ir_version": m.ir_version,
        "input_size": args.size,
        "sha256": sha256(args.out),
        "bytes": os.path.getsize(args.out),
        "node_count": len(m.graph.node),
        "initializer_count": len(m.graph.initializer),
        "inputs": inputs,
        "outputs": outputs,
        "op_histogram": dict(sorted(hist.items())),
        "resize_node_count": hist.get("Resize", 0),
        "last_node_ops": last_nodes,
    }
    os.makedirs(os.path.dirname(args.info), exist_ok=True)
    with open(args.info, "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
