#!/usr/bin/env python3
"""Explicit ONNX opset conversion for deployment preparation.

This is a standalone, user-initiated tool. VIE never calls it from Runtime,
Backend or InferenceEngine: converting a model is always a deliberate, separate
step so that a user's original ONNX is never silently rewritten (Phase 9.3.4
constraint 2.2). The classic use case is downgrading an opset that carries an
op the target ONNX Runtime CUDA EP has no kernel for -- e.g. YOLOv8n opset 20
(Resize v19) to opset 18 (Resize v13-v18) on ORT 1.20.1 -- but nothing here is
specific to YOLO, Resize, opset 18 or any single model.

Safety contract:
  * the input file is opened read-only and is never modified;
  * output == input is refused;
  * an existing output is refused unless --overwrite is given;
  * original and converted SHA256, original/target opset and converter version
    are always recorded;
  * a hard failure returns a non-zero exit code and is never reported as success.

Exit codes:
  0  success (conversion + onnx.checker + shape inference all clean)
  2  usage / safety error (bad args, missing input, output==input, exists+no-overwrite)
  3  conversion itself failed (e.g. version_converter cannot convert an op)
  4  converted, but onnx.checker.check_model() failed
  5  converted and checker passed, but shape inference failed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CONVERSION_FAILED = 3
EXIT_CHECKER_FAILED = 4
EXIT_SHAPE_INFER_FAILED = 5


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def opset_map(model) -> dict:
    """Domain -> opset version for every opset_import entry."""
    result = {}
    for import_entry in model.opset_import:
        domain = import_entry.domain or "ai.onnx"
        result[domain] = import_entry.version
    return result


def op_histogram(model) -> dict:
    return dict(Counter(node.op_type for node in model.graph.node))


def node_count(model) -> int:
    return len(model.graph.node)


def _emit(payload: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("ONNX opset conversion summary")
    print("-----------------------------")
    for key in (
        "status", "input_path", "output_path", "input_sha256", "output_sha256",
        "input_opset", "target_opset", "output_opset", "converter_version",
        "ir_version", "node_count_before", "node_count_after",
        "checker_passed", "shape_inference_passed", "error",
    ):
        if key in payload:
            print(f"  {key}: {payload[key]}")
    if payload.get("status") == "success":
        print("  RESULT: SUCCESS")


def run_conversion(args) -> int:
    # Lazy import so that --help / safety errors work even without onnx present.
    try:
        import onnx
        from onnx import version_converter
    except Exception as error:  # noqa: BLE001 - dependency availability is the point
        print(f"error: python onnx package unavailable ({error})", file=sys.stderr)
        print("hint: install with `pip install 'onnx>=1.14'` (onnx.version_converter).",
              file=sys.stderr)
        return EXIT_USAGE

    payload = {
        "schema_version": "1.0",
        "status": "not_run",
        "input_path": os.path.abspath(args.input),
        "output_path": os.path.abspath(args.output),
        "target_opset": args.target_opset,
        "overwrite": bool(args.overwrite),
        "converter_version": getattr(onnx, "__version__", "unknown"),
    }

    # --- safety gate: input must exist and be readable, output must be allowed.
    if not os.path.isfile(args.input):
        payload["status"] = "error_input_missing"
        payload["error"] = f"input file not found: {args.input}"
        _emit(payload, args.json)
        return EXIT_USAGE
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        payload["status"] = "error_output_equals_input"
        payload["error"] = "output path equals input path; refusing to modify the original model"
        _emit(payload, args.json)
        return EXIT_USAGE
    if os.path.exists(args.output) and not args.overwrite:
        payload["status"] = "error_output_exists"
        payload["error"] = f"output exists: {args.output} (pass --overwrite to replace)"
        _emit(payload, args.json)
        return EXIT_USAGE
    if args.target_opset < 1:
        payload["status"] = "error_invalid_opset"
        payload["error"] = f"target opset must be >= 1, got {args.target_opset}"
        _emit(payload, args.json)
        return EXIT_USAGE

    payload["input_sha256"] = sha256_file(args.input)

    try:
        model = onnx.load(args.input)
    except Exception as error:  # noqa: BLE001
        payload["status"] = "error_input_unreadable"
        payload["error"] = f"cannot parse input ONNX: {error}"
        _emit(payload, args.json)
        return EXIT_CONVERSION_FAILED

    payload["input_opset"] = opset_map(model)
    payload["ir_version"] = model.ir_version
    payload["node_count_before"] = node_count(model)
    payload["op_histogram_before"] = op_histogram(model)

    # --- conversion. version_converter raises if any op cannot be converted.
    try:
        converted = version_converter.convert_version(model, args.target_opset)
    except Exception as error:  # noqa: BLE001
        payload["status"] = "conversion_failed"
        payload["error"] = f"onnx.version_converter.convert_version failed: {error}"
        _emit(payload, args.json)
        return EXIT_CONVERSION_FAILED

    payload["output_opset"] = opset_map(converted)
    payload["node_count_after"] = node_count(converted)
    payload["op_histogram_after"] = op_histogram(converted)

    # --- write the converted model (original untouched; we only ever write output).
    try:
        onnx.save(converted, args.output)
        payload["output_sha256"] = sha256_file(args.output)
    except Exception as error:  # noqa: BLE001
        payload["status"] = "conversion_failed"
        payload["error"] = f"failed writing converted model: {error}"
        _emit(payload, args.json)
        return EXIT_CONVERSION_FAILED

    # --- structural validation of the produced model.
    checker_passed = False
    try:
        onnx.checker.check_model(converted)
        checker_passed = True
    except Exception as error:  # noqa: BLE001
        payload["checker_error"] = str(error)
    payload["checker_passed"] = checker_passed

    shape_inference_passed = False
    try:
        # strict_mode=False mirrors real-world inference: report failure only if
        # the inferencer itself cannot run, not for every unresolved dim.
        onnx.shape_inference.infer_shapes(converted, strict_mode=False)
        shape_inference_passed = True
    except Exception as error:  # noqa: BLE001
        payload["shape_inference_error"] = str(error)
    payload["shape_inference_passed"] = shape_inference_passed

    if not checker_passed:
        payload["status"] = "converted_check_failed"
        _emit(payload, args.json)
        return EXIT_CHECKER_FAILED
    if not shape_inference_passed:
        payload["status"] = "converted_shape_infer_failed"
        _emit(payload, args.json)
        return EXIT_SHAPE_INFER_FAILED

    payload["status"] = "success"
    _emit(payload, args.json)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicitly convert an ONNX model's opset (never modifies the input).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="source ONNX model (read-only)")
    parser.add_argument("--output", required=True, help="destination ONNX model")
    parser.add_argument("--target-opset", required=True, type=int,
                        help="target ai.onnx opset version, e.g. 18")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing --output file")
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_conversion(args)


if __name__ == "__main__":
    sys.exit(main())
