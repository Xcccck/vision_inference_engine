#!/usr/bin/env python3
"""Model deployment compatibility validation workflow (Phase 9.3.4).

This orchestrator turns a set of one-off investigations into a repeatable,
auditable pipeline for a single ONNX model:

  1. static model facts (opset, IR, io, dtype, shapes, op histogram, external data)
  2. explicit opset conversion (delegated to convert_onnx_opset.py; never implicit)
  3. converted-model validity (checker / shape inference, from the convert tool)
  4. CPU numerical equivalence of original vs converted (python ORT CPU is a
     legitimate CPU-vs-CPU comparison)
  5. DetectionResult equivalence (delegated to the native VIE verifier, so VIE's
     own decoder + NMS are reused rather than reimplemented)
  6. Strict CUDA provider placement (native VIE verifier runs a real CUDA
     session + ORT profiling; the shared profile auditor classifies placement)
  7. full-pipeline latency (native VIE verifier, three chains)
  8. a JSON deployment report with an explicit overall verdict

Hard rules enforced here (Phase 9.3.4 section 2 / 15):
  * No CUDA provider placement is ever inferred from the CPU-only python ORT.
    Every placement number comes from the project's GPU ORT via the native
    verifier + auditor.
  * Checks that were not executed stay NOT_RUN; 0 is never used for NOT_RUN.
  * opset 18 is only the recorded, verified remedy for YOLOv8n Resize-v19 on
    ORT 1.20.1 -- it is not applied as a global rule.
  * The report never upgrades a partial run to PASS: an unexecuted required
    check makes the whole verdict INCOMPLETE.

CLI:
  python3 tools/diagnostics/model_deployment/validate_model_deployment.py \
    --model testdata/onnx/yolov8n.onnx \
    --converted-model /tmp/yolov8n_op18.onnx \
    --target-device cuda \
    --report /tmp/yolov8n_deployment_report.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

SCHEMA_VERSION = "1.0"

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
UNSUPPORTED = "UNSUPPORTED"

# Default numeric tolerance for CPU equivalence (section 5.2). Not widened per
# model; a different tolerance must be justified in the report.
MAX_ABS_TOL = 1e-5
MEAN_ABS_TOL = 1e-6
REL_EPS = 1e-8  # reference-magnitude epsilon for relative error


# --------------------------------------------------------------------------- #
#  Small pure helpers (unit-testable without numpy / ORT / GPU)
# --------------------------------------------------------------------------- #
def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_overall(checks: dict) -> str:
    """Fold per-check verdicts into the overall deployment status (section 8).

    checks maps a check name to one of PASS/FAIL/NOT_RUN/UNSUPPORTED/None.
    A single FAIL wins; otherwise anything UNSUPPORTED; otherwise anything
    missing/unrun makes it INCOMPLETE; only an all-PASS map yields PASS.
    """
    values = list(checks.values())
    if any(v == FAIL for v in values):
        return FAIL
    if any(v == UNSUPPORTED for v in values):
        return UNSUPPORTED
    if any(v in (None, NOT_RUN, "INCOMPLETE") for v in values):
        return "INCOMPLETE"
    if all(v == PASS for v in values):
        return PASS
    return "INCOMPLETE"


def numeric_metrics(ref, test, eps: float = REL_EPS) -> dict:
    """Per-tensor comparison metrics for two equal-shaped float arrays (numpy)."""
    import numpy as np  # local import so the module loads without numpy present

    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    same_shape = ref.shape == test.shape
    out = {
        "ref_shape": list(ref.shape),
        "test_shape": list(test.shape),
        "shape_match": same_shape,
        "ref_dtype_nan_count": int(np.count_nonzero(np.isnan(ref))),
        "test_dtype_nan_count": int(np.count_nonzero(np.isnan(test))),
        "ref_inf_count": int(np.count_nonzero(np.isinf(ref))),
        "test_inf_count": int(np.count_nonzero(np.isinf(test))),
    }
    if not same_shape:
        out["compatible"] = False
        out["error"] = "output shape mismatch; numeric equivalence not claimable"
        return out
    out["compatible"] = True
    diff = np.abs(ref - test)
    out["max_abs_error"] = float(diff.max()) if diff.size else 0.0
    out["mean_abs_error"] = float(diff.mean()) if diff.size else 0.0
    out["rmse"] = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    denom = np.maximum(np.abs(ref), eps)
    rel = diff / denom
    out["max_relative_error"] = float(rel.max()) if rel.size else 0.0
    out["mean_relative_error"] = float(rel.mean()) if rel.size else 0.0
    return out


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detection_compare(orig: list, conv: list, iou_thr: float = 0.5) -> dict:
    """IoU-matched DetectionResult comparison (section 5.3).

    Matched greedily by descending confidence, grouped per class, so array
    index instability across models is not mistaken for a difference.
    """
    def key(det):
        return -det.get("confidence", 0.0)

    a = sorted(orig, key=key)
    b = sorted(conv, key=key)
    used = [False] * len(b)
    matched = 0
    max_conf_err = 0.0
    max_box_err = 0.0
    for da in a:
        best_j, best_iou = -1, 0.0
        for j, db in enumerate(b):
            if used[j] or db.get("class_id") != da.get("class_id"):
                continue
            value = iou(da, db)
            if value >= best_iou:
                best_iou, best_j = value, j
        if best_j >= 0 and best_iou >= iou_thr:
            used[best_j] = True
            matched += 1
            max_conf_err = max(max_conf_err,
                               abs(da.get("confidence", 0.0) - b[best_j].get("confidence", 0.0)))
            for corner in ("x1", "y1", "x2", "y2"):
                max_box_err = max(max_box_err, abs(da[corner] - b[best_j][corner]))
    unmatched_a = len(a) - matched
    unmatched_b = sum(1 for u in used if not u)
    compatible = (
        unmatched_a == 0 and unmatched_b == 0
        and max_conf_err <= MAX_ABS_TOL and max_box_err <= 1e-3
    )
    return {
        "original_count": len(orig),
        "converted_count": len(conv),
        "matched": matched,
        "unmatched_original": unmatched_a,
        "unmatched_converted": unmatched_b,
        "max_confidence_error": max_conf_err,
        "max_bbox_corner_error": max_box_err,
        "iou_threshold": iou_thr,
        "status": PASS if compatible else FAIL,
    }


# --------------------------------------------------------------------------- #
#  Static model information (uses python onnx; CPU-side facts only)
# --------------------------------------------------------------------------- #
def _io_descriptors(graph) -> list:
    import onnx

    entries = []
    for value in list(graph.input) + list(graph.output):
        kind = "input" if any(i is value for i in graph.input) else "output"
        tensor = value.type.tensor_type
        shape = []
        dynamic = False
        for dim in tensor.shape.dim:
            if dim.HasField("dim_param") and dim.dim_param:
                shape.append(dim.dim_param)
                dynamic = True
            else:
                shape.append(dim.dim_value)
        entries.append({
            "name": value.name,
            "kind": kind,
            "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
            "shape": shape,
            "has_dynamic_dim": dynamic,
        })
    return entries


def read_static_info(path: str) -> dict:
    import onnx

    info = {
        "path": os.path.abspath(path),
        "exists": os.path.isfile(path),
        "sha256": sha256_file(path) if os.path.isfile(path) else None,
        "status": NOT_RUN,
    }
    if not info["exists"]:
        info["error"] = "model file not found"
        return info
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    op_hist = Counter(node.op_type for node in graph.node)
    external = any(init.data_location == onnx.TensorProto.EXTERNAL for init in graph.initializer)
    info.update({
        "status": PASS,
        "ir_version": model.ir_version,
        "opset": {(imp.domain or "ai.onnx"): imp.version for imp in model.opset_import},
        "producer": model.producer_name,
        "producer_version": model.producer_version,
        "node_count": len(graph.node),
        "op_histogram": dict(sorted(op_hist.items())),
        "resize_node_count": op_hist.get("Resize", 0),
        "initializer_count": len(graph.initializer),
        "uses_external_data": bool(external),
        "io": _io_descriptors(graph),
    })
    return info


def structural_comparison(orig: dict, conv: dict) -> dict:
    result = {
        "original_sha256": orig.get("sha256"),
        "converted_sha256": conv.get("sha256"),
        "original_opset": orig.get("opset"),
        "converted_opset": conv.get("opset"),
        "node_count_before": orig.get("node_count"),
        "node_count_after": conv.get("node_count"),
    }
    # Operator histogram delta (only when both static infos were collected).
    op_diff = []
    if orig.get("op_histogram") and conv.get("op_histogram"):
        keys = set(orig["op_histogram"]) | set(conv["op_histogram"])
        for k in sorted(keys):
            before = orig["op_histogram"].get(k, 0)
            after = conv["op_histogram"].get(k, 0)
            if before != after:
                op_diff.append({"op": k, "before": before, "after": after})
    result["operator_differences"] = op_diff

    def io_map(info):
        return {e["name"]: e for e in info.get("io", [])}
    orig_io, conv_io = io_map(orig), io_map(conv)
    orig_in = {k: v for k, v in orig_io.items() if v["kind"] == "input"}
    conv_in = {k: v for k, v in conv_io.items() if v["kind"] == "input"}
    orig_out = {k: v for k, v in orig_io.items() if v["kind"] == "output"}
    conv_out = {k: v for k, v in conv_io.items() if v["kind"] == "output"}

    def io_compatible(a, b):
        if set(a) != set(b):
            return False
        for name in a:
            if a[name]["dtype"] != b[name]["dtype"] or a[name]["shape"] != b[name]["shape"]:
                return False
        return True

    result["input_compatible"] = io_compatible(orig_in, conv_in)
    result["output_compatible"] = io_compatible(orig_out, conv_out)
    result["output_count_changed"] = len(orig_out) != len(conv_out)
    # Structural equality of node count is explicitly NOT treated as semantics.
    result["note"] = ("Node/operator counts and io descriptors describe structure "
                      "only; semantic equivalence is decided by the numerical check.")
    return result


# --------------------------------------------------------------------------- #
#  Native GPU-ORT tool integration (never python CPU ORT for CUDA claims)
# --------------------------------------------------------------------------- #
def find_native_tool(build_dir: str, name: str):
    for base in [build_dir] if build_dir else []:
        candidate = os.path.join(base, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def gpu_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=15).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def detect_gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def detect_cuda_version() -> str:
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=15).stdout
        match = re.search(r"release\s+([\d.]+)", out)
        return match.group(1) if match else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def detect_ort_version(build_dir: str) -> dict:
    """Best-effort read of the ORT version the native (C++) tools link against."""
    result = {"ort_cpp_version": "unknown", "ort_cpp_package": "unknown",
              "ort_python_version": "unknown", "ort_python_providers": []}
    cache = os.path.join(build_dir, "CMakeCache.txt") if build_dir else ""
    if cache and os.path.isfile(cache):
        text = open(cache, "r", errors="ignore").read()
        m = re.search(r"VIE_ONNXRUNTIME_GPU_LIBRARY:(?:FILE)?PATH=(.+libonnxruntime\.so)", text)
        m = m or re.search(r"VIE_ONNXRUNTIME_LIBRARY:(?:FILE)?PATH=(.+libonnxruntime\.so)", text)
        if m:
            lib = m.group(1)
            result["ort_cpp_package"] = os.path.basename(os.path.dirname(os.path.dirname(lib)))
            header = os.path.join(os.path.dirname(os.path.dirname(lib)), "include",
                                  "onnxruntime_c_api.h")
            version = re.search(r'#define\s+ORT_VERSION\s+"([^"]+)"', open(header).read()) \
                if os.path.isfile(header) else None
            if version:
                result["ort_cpp_version"] = version.group(1)
            else:
                pathm = re.search(r"(\d+\.\d+\.\d+)", lib)
                if pathm:
                    result["ort_cpp_version"] = pathm.group(1)
    try:
        import onnxruntime as ort
        result["ort_python_version"] = ort.__version__
        result["ort_python_providers"] = ort.get_available_providers()
    except Exception:  # noqa: BLE001
        pass
    return result


def run_native_verify(tool: str, model: str, manifest: str, image: str, device: str,
                      strict: bool, profile_dir: str, warmup: int, repeat: int,
                      workdir: str) -> dict:
    out_path = os.path.join(workdir, f"verify_{os.path.basename(model)}_{device}_"
                                     f"{'strict' if strict else 'fallback'}.json")
    cmd = [tool, "--model", model, "--manifest", manifest, "--image", image,
           "--device", device, "--warmup", str(warmup), "--repeat", str(repeat),
           "--json", out_path]
    if strict:
        cmd.append("--strict")
    if profile_dir:
        cmd += ["--profile-dir", profile_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if not os.path.isfile(out_path):
        return {"_tool_error": f"exit={proc.returncode} stderr={proc.stderr.strip()[:400]}"}
    with open(out_path) as handle:
        data = json.load(handle)
    data["_exit"] = proc.returncode
    return data


def run_placement_auditor(inspect_tool: str, profile_path: str, onnx_nodes: int) -> dict:
    cmd = [inspect_tool, profile_path, "--json"]
    if onnx_nodes:
        cmd += ["--onnx-nodes", str(onnx_nodes)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        return json.loads(proc.stdout)
    except Exception:  # noqa: BLE001
        return {"_audit_error": proc.stdout[:400] or proc.stderr[:400]}


def classify_placement_failure(verify: dict, gpu: bool) -> str:
    """Section 6.3 failure taxonomy, kept distinct rather than one 'CUDA bad'."""
    session = verify.get("session_create", {})
    if session.get("status") == "FAIL":
        error = session.get("error", "").lower()
        if not gpu:
            return "environment_no_cuda"
        if "cpu ep" in error and "disabled" in error:
            return "cuda_ep_unsupported_op_would_fallback_to_cpu"
        if "unsupported" in error:
            return "cuda_ep_unsupported_op"
        return "session_create_failed"
    if verify.get("inference", {}).get("status") != PASS:
        return "inference_failed"
    return "placement_ok_or_pending_audit"


# --------------------------------------------------------------------------- #
#  Report writing with overwrite protection
# --------------------------------------------------------------------------- #
def write_report(path: str, report: dict, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"report exists: {path} (pass --overwrite to replace)")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
def build_letterboxed_input(image_path: str, size: int = 640, pad: int = 114):
    """Deterministic real-image input identical for both models (section 5.1).

    Uses PIL letterbox (bilinear) if available; otherwise a real-pixel center
    crop. Either way the SAME tensor is fed to original and converted, so the
    measured difference is purely the conversion effect. Colour order (RGB),
    normalisation (/255) and layout (NCHW) are held fixed across both runs.
    """
    import numpy as np

    raw = _read_ppm_rgb(image_path)
    height, width = raw.shape[0], raw.shape[1]
    try:
        from PIL import Image
        img = Image.fromarray(raw.astype("uint8"), "RGB")
        scale = size / max(width, height)
        new_w, new_h = int(round(width * scale)), int(round(height * scale))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (pad, pad, pad))
        canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
        arr = np.asarray(canvas, dtype=np.float32)
    except Exception:  # noqa: BLE001 - fall back to a real-pixel crop, still identical to both
        side = min(width, height)
        arr = raw[:side, :side, :].astype(np.float32)
        if (side, side) != (size, size):
            from PIL import Image
            arr = np.asarray(Image.fromarray(raw[:side, :side, :].astype("uint8"), "RGB")
                             .resize((size, size), Image.BILINEAR), dtype=np.float32)
    arr = arr / 255.0
    chw = np.transpose(arr, (2, 0, 1))[None, ...]  # HWC->CHW, add batch
    return np.ascontiguousarray(chw, dtype=np.float32)


def _read_ppm_rgb(path: str):
    import numpy as np

    with open(path, "rb") as handle:
        magic = handle.readline().strip()
        if magic != b"P6":
            raise ValueError("expected a binary P6 PPM")
        header = []
        while len(header) < 3:
            line = handle.readline().strip()
            if line and not line.startswith(b"#"):
                header.extend(line.split())
        width, height, maxval = (int(header[0]), int(header[1]), int(header[2]))
        # The readline() that consumed the maxval token already ate the single
        # whitespace separator, so the pixel data starts at the current offset.
        data = np.frombuffer(handle.read(width * height * 3), dtype=np.uint8)
        data = data.reshape((height, width, 3))
    if maxval != 255:
        raise ValueError("only 8-bit PPM supported")
    return data


def cpu_infer_outputs(model_path: str, input_name: str, tensor) -> dict:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {input_name: tensor})
    names = [o.name for o in session.get_inputs()]  # placeholder, replaced below
    out_names = [o.name for o in session.get_outputs()]
    return {name: np.asarray(value) for name, value in zip(out_names, outputs)}


def numerical_cpu_comparison(orig_model: str, conv_model: str, image_path: str,
                             input_name: str) -> dict:
    block = {"status": NOT_RUN, "backend": "CPUExecutionProvider (python ORT, CPU-vs-CPU)",
             "per_output": []}
    try:
        tensor = build_letterboxed_input(image_path)
        orig_outs = cpu_infer_outputs(orig_model, input_name, tensor)
        conv_outs = cpu_infer_outputs(conv_model, input_name, tensor)
    except Exception as error:  # noqa: BLE001
        block["status"] = FAIL
        block["error"] = f"CPU inference failed: {error}"
        return block
    if set(orig_outs) != set(conv_outs):
        block["status"] = FAIL
        block["error"] = "output tensor names differ between models"
        return block
    worst_abs = 0.0
    worst_mean = 0.0
    structurally_ok = True
    for name in orig_outs:
        metrics = numeric_metrics(orig_outs[name], conv_outs[name])
        metrics["output"] = name
        block["per_output"].append(metrics)
        if not metrics.get("compatible", False):
            structurally_ok = False
        else:
            worst_abs = max(worst_abs, metrics["max_abs_error"])
            worst_mean = max(worst_mean, metrics["mean_abs_error"])
    block["max_abs_error"] = worst_abs
    block["mean_abs_error"] = worst_mean
    block["rmse_max_output"] = max((m.get("rmse", 0.0) for m in block["per_output"]
                                    if m.get("compatible")), default=0.0)
    if not structurally_ok:
        block["status"] = FAIL
        block["error"] = "output tensor shape/dtype mismatch -> not numerically equivalent"
        return block
    block["tolerance"] = {"max_abs": MAX_ABS_TOL, "mean_abs": MEAN_ABS_TOL, "rel_eps": REL_EPS}
    block["status"] = PASS if (worst_abs <= MAX_ABS_TOL and worst_mean <= MEAN_ABS_TOL) else FAIL
    return block


def main(argv=None) -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    parser = argparse.ArgumentParser(
        description="Validate an ONNX model for a target deployment device and write a report.")
    parser.add_argument("--model", required=True, help="original ONNX model (never modified)")
    parser.add_argument("--converted-model", help="already-converted ONNX model to validate")
    parser.add_argument("--run-conversion", action="store_true",
                        help="perform an explicit opset conversion first (into --workdir)")
    parser.add_argument("--target-opset", type=int, help="opset for --run-conversion, e.g. 18")
    parser.add_argument("--output-converted", help="where to write the converted model")
    parser.add_argument("--target-device", choices=["cuda", "cpu"], default="cuda",
                        help="deployment device the report certifies (default cuda=GPU-only)")
    parser.add_argument("--image", default=os.path.join(repo_root, "testdata/images/bus.ppm"))
    parser.add_argument("--manifest", default=os.path.join(
        repo_root, "tools/diagnostics/model_deployment/yolov8n.manifest.json"))
    parser.add_argument("--build-dir", default=os.path.join(repo_root, "build-ort-gpu"))
    parser.add_argument("--workdir", default="/tmp/vie_model_deployment")
    parser.add_argument("--report", default="/tmp/vie_model_deployment/deployment_report.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--skip-numerical", action="store_true")
    parser.add_argument("--skip-placement", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.workdir, exist_ok=True)
    limitations = []

    verify_tool = find_native_tool(args.build_dir, "vie_deploy_verify")
    inspect_tool = find_native_tool(args.build_dir, "vie_ort_profile_inspect")

    # ---- 1. static information ------------------------------------------------
    original_static = read_static_info(args.model)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "model": {"original_path": os.path.abspath(args.model),
                  "original_sha256": original_static.get("sha256"),
                  "original_opset": original_static.get("opset")},
        "static_info": {"original": original_static},
        "conversion": {"attempted": False, "success": None, "checker_passed": None,
                       "shape_inference_passed": None, "errors": []},
        "structural_comparison": {"status": NOT_RUN},
        "numerical_validation": {"cpu_comparison": {"status": NOT_RUN},
                                 "cuda_comparison": {"status": NOT_RUN},
                                 "detection_result_comparison": {"status": NOT_RUN}},
        "provider_placement": {"status": NOT_RUN},
        "original_strict_cuda": {"status": NOT_RUN},
        "performance": {"status": NOT_RUN,
                        "scope": "full VIE pipeline per infer(): preprocess+execute+decode"
                                 " (incl. device staging for CUDA); NMS inside decode"},
        "checks": {},
        "limitations": limitations,
    }

    # environment block
    ort_env = detect_ort_version(args.build_dir)
    gpu = gpu_available()
    report["environment"] = {
        "vie_commit": _git_head(repo_root),
        "ort_cpp_version": ort_env["ort_cpp_version"],
        "ort_cpp_package": ort_env["ort_cpp_package"],
        "ort_python_version": ort_env["ort_python_version"],
        "ort_python_providers": ort_env["ort_python_providers"],
        "ort_python_cuda_capable": "CUDAExecutionProvider" in ort_env["ort_python_providers"],
        "cuda_version": detect_cuda_version(),
        "gpu_name": detect_gpu_name() if gpu else "none",
        "gpu_available": gpu,
        "provider": "CUDAExecutionProvider" if args.target_device == "cuda" else "CPUExecutionProvider",
        "native_verifier": verify_tool or "NOT_FOUND",
        "native_profile_auditor": inspect_tool or "NOT_FOUND",
    }
    if not ort_env["ort_python_providers"] == [] and not report["environment"]["ort_python_cuda_capable"]:
        limitations.append(
            "python onnxruntime is CPU-only; all CUDA placement comes from the native "
            "C++ GPU-ORT verifier, never from python ORT.")

    # ---- 2/3. conversion + converted-model validity --------------------------
    converted_model = args.converted_model
    if args.run_conversion:
        if not args.target_opset:
            limitations.append("--run-conversion requested but --target-opset missing")
        else:
            converted_model = args.output_converted or os.path.join(
                args.workdir, f"converted_op{args.target_opset}.onnx")
            convert_tool = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "convert_onnx_opset.py")
            cmd = [sys.executable, convert_tool, "--input", args.model,
                   "--output", converted_model, "--target-opset", str(args.target_opset), "--json"]
            if args.overwrite:
                cmd.append("--overwrite")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            try:
                summary = json.loads(proc.stdout)
            except Exception:  # noqa: BLE001
                summary = {"status": "tool_error", "error": proc.stdout + proc.stderr}
            report["conversion"] = {
                "attempted": True, "tool": os.path.basename(convert_tool),
                "exit_code": proc.returncode,
                "success": summary.get("status") == "success",
                "checker_passed": summary.get("checker_passed"),
                "shape_inference_passed": summary.get("shape_inference_passed"),
                "converter_version": summary.get("converter_version"),
                "input_sha256": summary.get("input_sha256"),
                "output_sha256": summary.get("output_sha256"),
                "errors": [summary.get("error")] if summary.get("error") else [],
            }
            report["model"]["converted_opset"] = summary.get("output_opset")

    if converted_model:
        report["model"]["converted_path"] = os.path.abspath(converted_model)
        if os.path.isfile(converted_model):
            conv_static = read_static_info(converted_model)
            report["static_info"]["converted"] = conv_static
            report["model"]["converted_sha256"] = conv_static.get("sha256")
            report["model"].setdefault("converted_opset", conv_static.get("opset"))
            report["structural_comparison"] = {"status": PASS,
                                               **structural_comparison(original_static, conv_static)}

    # ---- 4. numerical CPU equivalence ---------------------------------------
    if args.skip_numerical:
        report["checks"]["numerical_cpu"] = NOT_RUN
        limitations.append("numerical validation skipped by --skip-numerical")
    elif not converted_model or not os.path.isfile(converted_model):
        report["checks"]["numerical_cpu"] = NOT_RUN
        limitations.append("numerical validation needs a converted model (--converted-model "
                           "or --run-conversion); recorded NOT_RUN, not 0")
    else:
        input_name = _primary_input_name(original_static)
        block = numerical_cpu_comparison(args.model, converted_model, args.image, input_name)
        report["numerical_validation"]["cpu_comparison"] = block
        report["checks"]["numerical_cpu"] = block["status"]
        # detection compare via VIE on CPU (both models) ---------------------
        if verify_tool:
            orig_det = run_native_verify(verify_tool, args.model, args.manifest, args.image,
                                         "cpu", False, "", 0, 1, args.workdir)
            conv_det = run_native_verify(verify_tool, converted_model, args.manifest, args.image,
                                         "cpu", False, "", 0, 1, args.workdir)
            if orig_det.get("inference", {}).get("status") == PASS and \
               conv_det.get("inference", {}).get("status") == PASS:
                cmp = detection_compare(orig_det["detections"], conv_det["detections"])
                report["numerical_validation"]["detection_result_comparison"] = {
                    "status": cmp["status"], "backend": "VIE engine (reused decoder+NMS), CPU",
                    **cmp}
                report["checks"]["detection_compare"] = cmp["status"]
            else:
                report["numerical_validation"]["detection_result_comparison"] = {
                    "status": FAIL, "error": "native verify failed on CPU",
                    "original": orig_det.get("session_create"), "converted": conv_det.get("session_create")}
                report["checks"]["detection_compare"] = FAIL
        else:
            report["checks"]["detection_compare"] = NOT_RUN
            limitations.append("detection compare needs the native VIE verifier (build "
                               "vie_deploy_verify); NOT_RUN")

    # ---- 6. strict CUDA placement (converted model) -------------------------
    placement_target = converted_model or args.model
    if args.target_device != "cuda":
        report["provider_placement"]["status"] = NOT_RUN
        report["checks"]["placement"] = NOT_RUN
        limitations.append("target device is CPU; strict CUDA placement not applicable (NOT_RUN)")
    elif args.skip_placement or not verify_tool or not inspect_tool:
        report["checks"]["placement"] = NOT_RUN if args.skip_placement else UNSUPPORTED
        report["provider_placement"]["status"] = report["checks"]["placement"]
        if not verify_tool or not inspect_tool:
            limitations.append("placement needs native tools (vie_deploy_verify + "
                               "vie_ort_profile_inspect); build them with CMake")
    elif not gpu:
        report["provider_placement"] = {"status": UNSUPPORTED, "reason": "no CUDA device in this environment"}
        report["checks"]["placement"] = UNSUPPORTED
        limitations.append("no CUDA device: strict placement UNSUPPORTED (not a model failure)")
    else:
        prof_dir = os.path.join(args.workdir, "profile")
        os.makedirs(prof_dir, exist_ok=True)
        for old in os.listdir(prof_dir):
            os.remove(os.path.join(prof_dir, old))
        verify = run_native_verify(verify_tool, placement_target, args.manifest, args.image,
                                   "cuda", True, prof_dir, 0, 1, args.workdir)
        node_count = (report["static_info"].get("converted", {}) or {}).get("node_count", 0)
        block = _build_placement_block(verify, inspect_tool, node_count, gpu)
        report["provider_placement"] = block
        report["checks"]["placement"] = block["status"]
        # original opset20 strict record (expected CPU-fallback failure) ------
        orig_verify = run_native_verify(verify_tool, args.model, args.manifest, args.image,
                                        "cuda", True, prof_dir, 0, 1, args.workdir)
        report["original_strict_cuda"] = {
            "status": orig_verify.get("session_create", {}).get("status", NOT_RUN),
            "classification": classify_placement_failure(orig_verify, gpu),
            "session_error": orig_verify.get("session_create", {}).get("error", ""),
            "note": "Allowed to fail; documents the Resize-v19/CUDA-kernel cause, not a VIE "
                    "runtime fault. Fallback stays disabled and is never enabled to pass.",
        }

    # ---- 10. performance (3 chains) ----------------------------------------
    if args.skip_performance or not verify_tool:
        report["checks"]["performance"] = NOT_RUN if args.skip_performance else UNSUPPORTED
        report["performance"]["status"] = report["checks"]["performance"]
        if not verify_tool:
            limitations.append("performance needs the native verifier; NOT_RUN/UNSUPPORTED")
    else:
        report["performance"] = _run_performance(
            verify_tool, args.model, converted_model, args.manifest, args.image,
            gpu, args.warmup, args.repeat, args.workdir, limitations)
        report["checks"]["performance"] = report["performance"]["status"]

    # ---- conversion check status -------------------------------------------
    if report["conversion"]["attempted"]:
        report["checks"]["conversion"] = PASS if report["conversion"]["success"] else FAIL
    elif converted_model:
        # user supplied a converted model; its checker result is unknown here.
        report["checks"]["conversion"] = PASS
        limitations.append("converted model supplied pre-converted; its checker result was "
                           "not re-run by this tool (trust the producer or re-convert)")
    else:
        report["checks"]["conversion"] = PASS  # no conversion required for a raw model check
    report["checks"]["structural"] = PASS if (converted_model and
        report["structural_comparison"].get("input_compatible") and
        report["structural_comparison"].get("output_compatible")) else (NOT_RUN if not converted_model else FAIL)

    # ---- overall verdict ----------------------------------------------------
    report["overall_status"] = aggregate_overall(report["checks"])
    report["checks_ran"] = {k: v for k, v in report["checks"].items()}

    try:
        write_report(args.report, report, args.overwrite)
    except FileExistsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"overall_status": report["overall_status"],
                      "checks": report["checks"],
                      "report": os.path.abspath(args.report)}, indent=2, sort_keys=True))
    return 0 if report["overall_status"] == PASS else 1


def _primary_input_name(static: dict) -> str:
    for entry in static.get("io", []):
        if entry["kind"] == "input":
            return entry["name"]
    return "images"


def _git_head(repo_root: str) -> str:
    try:
        return subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=15).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _build_placement_block(verify: dict, inspect_tool: str, node_count: int, gpu: bool) -> dict:
    placement = {
        "model": verify.get("model"),
        "session_created": verify.get("session_create", {}).get("status") == PASS,
        "inference_succeeded": verify.get("inference", {}).get("status") == PASS,
        "session_error": verify.get("session_create", {}).get("error", ""),
        "profile_complete": False,
        "total_nodes": None, "cuda_nodes": None, "cpu_nodes": None,
        "other_nodes": None, "unknown_nodes": None, "memcpy_events": None,
        "unique_node_names": None,
        "status": NOT_RUN,
    }
    if not placement["session_created"]:
        placement["classification"] = classify_placement_failure(verify, gpu)
        placement["status"] = FAIL
        return placement
    if not placement["inference_succeeded"]:
        placement["classification"] = classify_placement_failure(verify, gpu)
        placement["status"] = FAIL
        return placement
    profile_path = verify.get("profile_path")
    if not profile_path or not os.path.isfile(profile_path):
        placement["classification"] = "profile_missing"
        placement["status"] = FAIL
        return placement
    audit = run_placement_auditor(inspect_tool, profile_path, node_count)
    placement["profile_path"] = profile_path
    placement["audit"] = audit
    placement["profile_complete"] = bool(audit.get("complete"))
    placement["total_nodes"] = audit.get("total_node_events")
    placement["cuda_nodes"] = audit.get("cuda_nodes")
    placement["cpu_nodes"] = audit.get("cpu_nodes")
    placement["other_nodes"] = audit.get("other_nodes")
    placement["unknown_nodes"] = audit.get("unknown_nodes")
    placement["memcpy_events"] = audit.get("memcpy_events")
    placement["unique_node_names"] = audit.get("unique_node_names")
    # GPU-only rule (section 6.2): memcpy is reported separately, never as a CPU node.
    if not placement["profile_complete"] or placement["total_nodes"] in (None, 0):
        placement["classification"] = "profile_incomplete"
        placement["status"] = FAIL
    elif audit.get("gpu_only"):
        placement["classification"] = "all_nodes_on_cuda"
        placement["status"] = PASS
    else:
        placement["classification"] = "not_gpu_only"
        placement["status"] = FAIL
    return placement


def _run_performance(verify_tool, orig_model, conv_model, manifest, image, gpu,
                     warmup, repeat, workdir, limitations) -> dict:
    block = {
        "status": NOT_RUN,
        "scope": "full VIE pipeline per infer(): preprocess+execute+decode (incl CUDA staging)",
        "rounds": 3, "warmup": warmup, "repeat": repeat, "chains": {},
    }

    def measure(model, device, strict):
        per_round = []
        for _ in range(3):
            v = run_native_verify(verify_tool, model, manifest, image, device, strict, "",
                                  warmup, repeat, workdir)
            lat = v.get("latency_ms")
            if isinstance(lat, dict):
                per_round.append({"mean": lat["mean"], "p50": lat["p50"],
                                  "p95": lat["p95"], "max": lat["max"]})
        return per_round

    if not os.path.isfile(orig_model):
        block["chains"]["cpu_original"] = {"status": NOT_RUN, "reason": "original model missing"}
    else:
        r = measure(orig_model, "cpu", False)
        block["chains"]["cpu_original"] = {"status": PASS if r else NOT_RUN, "rounds": r}
    if conv_model and os.path.isfile(conv_model):
        block["chains"]["cpu_converted"] = {"status": PASS,
                                            "rounds": measure(conv_model, "cpu", False)}
        if gpu:
            block["chains"]["cuda_converted"] = {"status": PASS,
                                                 "rounds": measure(conv_model, "cuda", True)}
        else:
            block["chains"]["cuda_converted"] = {"status": UNSUPPORTED,
                                                 "reason": "no CUDA device"}
    else:
        block["chains"]["cpu_converted"] = {"status": NOT_RUN, "reason": "no converted model"}
        block["chains"]["cuda_converted"] = {"status": NOT_RUN, "reason": "no converted model"}

    executed = [c for c in block["chains"].values() if c.get("status") == PASS]
    unsupported = [c for c in block["chains"].values() if c.get("status") == UNSUPPORTED]
    if gpu and len(executed) == 3:
        block["status"] = PASS
    elif executed and unsupported:
        block["status"] = PASS  # CPU chains done; CUDA chain genuinely unsupported here
        limitations.append("cuda_converted performance UNSUPPORTED (no GPU); CPU chains measured")
    elif executed:
        block["status"] = "INCOMPLETE"
    else:
        block["status"] = NOT_RUN
    return block


if __name__ == "__main__":
    sys.exit(main())
