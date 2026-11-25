#!/usr/bin/env python3
"""Phase 9.3.3: create an opset-downgraded COPY of a model to test whether the
YOLOv8n Resize nodes become CUDA-EP-eligible (ORT 1.20.1 registers a CUDA
Resize kernel for opset <= 18, but not for the Resize v19 schema used by
opset >= 19).

Read-only on the input: writes a new file, never touches the baseline.
Usage: python3 downgrade_opset.py --in MODEL.onnx --out MODEL_op18.onnx --opset 18
"""
import argparse

import onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    m = onnx.load(args.src)
    cur = m.opset_import[0].version
    print(f"input opset={cur}, converting to {args.opset} ...")
    try:
        conv = onnx.version_converter.convert_version(m, args.opset)
    except Exception as e:  # noqa
        print(f"convert_version FAILED: {e}")
        raise SystemExit(1)
    onnx.checker.check_model(conv)
    onnx.save(conv, args.dst)
    resizes = [n.name for n in conv.graph.node if n.op_type == "Resize"]
    print(f"wrote {args.dst} opset={conv.opset_import[0].version} "
          f"nodes={len(conv.graph.node)} Resize={resizes}")


if __name__ == "__main__":
    main()
