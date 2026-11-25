#!/usr/bin/env python3
"""Phase 9.3.3: build Resize models that change ONLY the ONNX opset, to test
whether the CUDA EP's Resize kernel registration is gated by opset/version
(see the binary evidence: CUDA Resize kernel code exists but the registry
lookup for standard `Resize` fails at opset 20).

Usage: python3 opset_sweep.py --outdir /tmp/opset
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSETS = [10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21]


def build(opset, attrs, in_dims, out_dims):
    sc = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), "sc")
    node = helper.make_node("Resize", ["X", "", "sc"], ["Y"], name="Resize", **attrs)
    g = helper.make_graph(
        [node], "r",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, in_dims)],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_dims)],
        initializer=[sc],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 9
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/opset")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    attrs = dict(mode="nearest", coordinate_transformation_mode="asymmetric",
                 nearest_mode="floor", cubic_coeff_a=-0.75)
    for op in OPSETS:
        m = build(op, attrs, [1, 256, 20, 20], [1, 256, 40, 40])
        path = os.path.join(args.outdir, f"resize_op{op}.onnx")
        onnx.save(m, path)
        try:
            onnx.checker.check_model(m)
            print(f"op{op}: saved {path} (checker=PASS)")
        except Exception as e:  # noqa
            print(f"op{op}: saved {path} (checker=NOTE {e})")


if __name__ == "__main__":
    main()
