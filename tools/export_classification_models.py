#!/usr/bin/env python3
"""Export torchvision classifiers for the optional VIE compatibility check.

Pinned to the versions already installed in this environment:
torch 2.14.1, torchvision 0.29.2, onnx 1.23.2. Weights stay at the module's
default initialization (weights=None). Nothing is downloaded, and the ONNX
files are written outside the repository.

The script does not infer ImageNet preprocessing. The comparison tensor is the
same uint8 pattern divided by 255, in NCHW. That is a test input, not a claim
that the model was trained with this normalization.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torchvision
from torch.nn import functional as F
from torchvision.models import efficientnet_b0, mobilenet_v2, mobilenet_v3_small, resnet18

EXPECTED = {
    "torch": "2.14.1",
    "torchvision": "0.29.2",
    "onnx": "1.23.2",
}


def require_versions():
    import onnx as onnx_module

    found = {
        "torch": torch.__version__.split("+")[0],
        "torchvision": torchvision.__version__.split("+")[0],
        "onnx": onnx_module.__version__,
    }
    for name, expected in EXPECTED.items():
        if found[name] != expected:
            raise SystemExit(f"{name} {found[name]} does not match the pinned {expected}")
    return found


def pattern(height, width):
    count = height * width * 3
    values = np.array([((index * 17 + 3) % 251) for index in range(count)], dtype=np.uint8)
    return values.reshape(height, width, 3)


def nchw(image):
    tensor = torch.from_numpy(image).to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.contiguous()


def export_logits(model, path, sample):
    model.eval()
    torch.onnx.export(
        model,
        sample,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )


def append_softmax(source, destination):
    model = onnx.load(str(source))
    graph = model.graph
    softmax = onnx.helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1)
    graph.node.append(softmax)
    output = graph.output[0]
    graph.output.pop()
    graph.output.append(
        onnx.helper.make_tensor_value_info("probabilities", output.type.tensor_type.elem_type, [1, 1000])
    )
    onnx.save(model, str(destination))


def top_indices(values, count):
    order = np.argsort(-values, kind="mergesort")
    return [(int(index), float(values[index])) for index in order[:count]]


def main():
    versions = require_versions()
    destination = Path("/tmp/vie_phase66")
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    image = pattern(224, 224)
    sample = nchw(image)
    builders = {
        "resnet18": resnet18,
        "mobilenet_v2": mobilenet_v2,
        "mobilenet_v3_small": mobilenet_v3_small,
        "efficientnet_b0": efficientnet_b0,
    }
    print("versions", versions)
    print("onnxruntime python is not installed; PyTorch is the reference and VIE's ORT session is compared later")
    for name, builder in builders.items():
        model = builder(weights=None).eval()
        path = destination / f"{name}_untrained_opset17.onnx"
        export_logits(model, path, sample)
        with torch.no_grad():
            pytorch = model(sample).detach().to(torch.float32).numpy().reshape(-1)
        pytorch.tofile(path.with_suffix(".pytorch.f32"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(
            f"{name} bytes={path.stat().st_size} sha256={digest} "
            f"top1={top_indices(pytorch, 5)}"
        )
        if name == "resnet18":
            softmax_path = destination / "resnet18_untrained_opset17_softmax.onnx"
            append_softmax(path, softmax_path)
            probabilities = F.softmax(torch.from_numpy(pytorch), dim=0).numpy().astype(np.float32)
            probabilities.tofile(softmax_path.with_suffix(".pytorch.f32"))
            softmax_digest = hashlib.sha256(softmax_path.read_bytes()).hexdigest()
            print(
                f"resnet18_softmax bytes={softmax_path.stat().st_size} sha256={softmax_digest} "
                f"pytorch_softmax_sum={float(probabilities.sum()):.8g} top1={top_indices(probabilities, 5)}"
            )


if __name__ == "__main__":
    sys.exit(main())
