#!/usr/bin/env python3
"""Tests for the Phase 9.3.4 model-deployment workflow.

Two tiers:
  * Pure / conversion tests -- no GPU, no native C++ tools, no real model. They
    exercise the decision logic, report writer, comparison maths and the
    convert_onnx_opset.py CLI (with a tiny generated model). These MUST pass in
    a CPU-only build environment (Phase 9.3.4 section 11).
  * Real-model tests -- run yolov8n opset20 -> opset18 through the native VIE
    verifier + profile auditor. These SKIP (never silently PASS) when the GPU,
    the built tools, or the fixtures are unavailable.

Run:  pytest -q tools/diagnostics/model_deployment/test_model_deployment.py
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CONVERT = os.path.join(HERE, "convert_onnx_opset.py")
VALIDATE = os.path.join(HERE, "validate_model_deployment.py")


def _load_validate_module():
    spec = importlib.util.spec_from_file_location("vmd", VALIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vmd = _load_validate_module()


def _tiny_model(tmp_path, opset=20):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    node = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", opset)],
                              ir_version=9)
    path = str(tmp_path / f"tiny_op{opset}.onnx")
    onnx.save(model, path)
    return path


def _run_convert(*args):
    return subprocess.run([sys.executable, CONVERT, *args], capture_output=True, text=True)


# --------------------------------------------------------------------------- #
#  Conversion tool (section 11: convert tool)
# --------------------------------------------------------------------------- #
def test_convert_missing_input(tmp_path):
    proc = _run_convert("--input", str(tmp_path / "nope.onnx"),
                        "--output", str(tmp_path / "out.onnx"), "--target-opset", "18")
    assert proc.returncode == vmd.__dict__.get("EXIT_USAGE", 2) or proc.returncode == 2
    assert "input file not found" in proc.stdout + proc.stderr


def test_convert_output_equals_input(tmp_path):
    model = _tiny_model(tmp_path)
    proc = _run_convert("--input", model, "--output", model, "--target-opset", "18")
    assert proc.returncode == 2
    assert "equals input" in proc.stdout + proc.stderr


def test_convert_success_and_sha(tmp_path):
    onnx = pytest.importorskip("onnx")
    model = _tiny_model(tmp_path, opset=20)
    out = str(tmp_path / "conv.onnx")
    proc = _run_convert("--input", model, "--output", out, "--target-opset", "18", "--json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["status"] == "success"
    assert summary["checker_passed"] is True
    assert summary["input_opset"]["ai.onnx"] == 20
    assert summary["output_opset"]["ai.onnx"] == 18
    # SHA256 values are real, distinct and match a fresh computation.
    assert summary["input_sha256"] == vmd.sha256_file(model)
    assert summary["output_sha256"] == vmd.sha256_file(out)


def test_convert_refuses_existing_output(tmp_path):
    model = _tiny_model(tmp_path)
    out = _tiny_model(tmp_path, opset=13)
    out = str(tmp_path / "tiny_op13.onnx")  # exists now
    proc = _run_convert("--input", model, "--output", out, "--target-opset", "18")
    assert proc.returncode == 2
    assert "output exists" in proc.stdout + proc.stderr


def test_convert_overwrite_allowed(tmp_path):
    model = _tiny_model(tmp_path)
    out = str(tmp_path / "conv.onnx")
    open(out, "w").close()  # pre-create
    proc = _run_convert("--input", model, "--output", out, "--target-opset", "18", "--overwrite")
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_convert_invalid_target_opset(tmp_path):
    model = _tiny_model(tmp_path)
    proc = _run_convert("--input", model, "--output", str(tmp_path / "c.onnx"),
                        "--target-opset", "9999")
    assert proc.returncode != 0
    assert "conversion_failed" in proc.stdout + proc.stderr or "invalid version" in proc.stdout


# --------------------------------------------------------------------------- #
#  Decision logic (section 11: compatibility verdict)
# --------------------------------------------------------------------------- #
def test_aggregate_all_pass():
    assert vmd.aggregate_overall({"a": "PASS", "b": "PASS"}) == "PASS"


def test_aggregate_any_fail_wins():
    assert vmd.aggregate_overall({"a": "PASS", "b": "FAIL", "c": None}) == "FAIL"


def test_aggregate_unsupported_beats_incomplete():
    assert vmd.aggregate_overall({"a": "PASS", "b": "UNSUPPORTED", "c": "NOT_RUN"}) == "UNSUPPORTED"


def test_aggregate_not_run_is_incomplete_not_pass():
    # Session/numeric/etc. not executed must never read as PASS.
    assert vmd.aggregate_overall({"a": "PASS", "b": "NOT_RUN"}) == "INCOMPLETE"
    assert vmd.aggregate_overall({"a": "PASS", "b": None}) == "INCOMPLETE"


def test_aggregate_single_nonpass_not_pass():
    for bad in ("FAIL", "UNSUPPORTED", "NOT_RUN", None, "INCOMPLETE"):
        assert vmd.aggregate_overall({"x": "PASS", "y": bad}) != "PASS"


# --------------------------------------------------------------------------- #
#  Comparison maths (sections 5.2 / 5.3)
# --------------------------------------------------------------------------- #
def test_numeric_metrics_identical():
    np = pytest.importorskip("numpy")
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    m = vmd.numeric_metrics(arr, arr.copy())
    assert m["compatible"] is True
    assert m["max_abs_error"] == 0.0
    assert m["rmse"] == 0.0


def test_numeric_metrics_shape_mismatch_not_comparable():
    np = pytest.importorskip("numpy")
    m = vmd.numeric_metrics(np.zeros((2, 2)), np.zeros((3, 3)))
    assert m["compatible"] is False
    # A shape mismatch must NOT be reported as an equivalence number; it is a
    # structural incompatibility.
    assert "max_abs_error" not in m


def test_detection_compare_identical_pass():
    dets = [{"class_id": 0, "confidence": 0.9, "x1": 1, "y1": 2, "x2": 30, "y2": 40}]
    result = vmd.detection_compare(dets, [dict(d) for d in dets])
    assert result["status"] == "PASS"
    assert result["matched"] == 1 and result["unmatched_original"] == 0


def test_detection_compare_missing_detection_fails():
    a = [{"class_id": 0, "confidence": 0.9, "x1": 1, "y1": 2, "x2": 30, "y2": 40},
         {"class_id": 2, "confidence": 0.7, "x1": 5, "y1": 5, "x2": 20, "y2": 20}]
    b = [a[0]]
    result = vmd.detection_compare(a, b)
    assert result["status"] == "FAIL"
    assert result["unmatched_original"] == 1


def test_iou_basic():
    box = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
    assert abs(vmd.iou(box, box) - 1.0) < 1e-6
    disjoint = {"x1": 100, "y1": 100, "x2": 110, "y2": 110}
    assert vmd.iou(box, disjoint) == 0.0


# --------------------------------------------------------------------------- #
#  Report writer (section 11: report generation)
# --------------------------------------------------------------------------- #
def test_write_report_overwrite_guard(tmp_path):
    path = str(tmp_path / "report.json")
    vmd.write_report(path, {"schema_version": "1.0", "overall_status": "PASS"}, overwrite=False)
    with pytest.raises(FileExistsError):
        vmd.write_report(path, {"schema_version": "1.0"}, overwrite=False)
    vmd.write_report(path, {"schema_version": "1.0", "overall_status": "FAIL"}, overwrite=True)
    assert json.load(open(path))["overall_status"] == "FAIL"


def test_static_info_missing_is_not_run_not_zero(tmp_path):
    info = vmd.read_static_info(str(tmp_path / "absent.onnx"))
    assert info["status"] == "NOT_RUN"
    assert info["sha256"] is None


# --------------------------------------------------------------------------- #
#  Real model (sections 9 / 11) -- skip when the environment cannot run them
# --------------------------------------------------------------------------- #
MODEL = os.path.join(REPO, "testdata/onnx/yolov8n.onnx")
IMAGE = os.path.join(REPO, "testdata/images/bus.ppm")
BUILD = os.path.join(REPO, "build-ort-gpu")
VERIFY = os.path.join(BUILD, "vie_deploy_verify")
INSPECT = os.path.join(BUILD, "vie_ort_profile_inspect")
MANIFEST = os.path.join(HERE, "yolov8n.manifest.json")

_gpu = vmd.gpu_available()
_tools = os.path.isfile(VERIFY) and os.path.isfile(INSPECT)
_cuda_only_skip = pytest.mark.skipif(
    not (_gpu and _tools and os.path.isfile(MODEL) and os.path.isfile(MANIFEST)),
    reason="requires GPU + built native VIE verifier + fixtures (skip, not pass)")


@_cuda_only_skip
def test_real_yolov8n_workflow_passes(tmp_path):
    report = str(tmp_path / "report.json")
    converted = str(tmp_path / "yolov8n_op18.onnx")
    proc = subprocess.run(
        [sys.executable, VALIDATE, "--model", MODEL,
         "--run-conversion", "--target-opset", "18", "--output-converted", converted,
         "--target-device", "cuda", "--build-dir", BUILD,
         "--workdir", str(tmp_path), "--report", report, "--overwrite",
         "--warmup", "3", "--repeat", "10"],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.load(open(report))
    assert data["overall_status"] == "PASS"
    assert data["numerical_validation"]["cpu_comparison"]["status"] == "PASS"
    assert data["numerical_validation"]["cpu_comparison"]["max_abs_error"] <= 1e-5
    assert data["numerical_validation"]["detection_result_comparison"]["status"] == "PASS"
    placement = data["provider_placement"]
    assert placement["status"] == "PASS"
    assert placement["cpu_nodes"] == 0 and placement["unknown_nodes"] == 0
    assert placement["cuda_nodes"] == placement["total_nodes"] and placement["cuda_nodes"] > 0


@_cuda_only_skip
def test_real_opset20_strict_cuda_is_recorded_as_expected_failure():
    # Original opset20 must FAIL strict CUDA with the CPU-fallback classification and
    # must NOT be re-labelled as a VIE runtime fault or hidden by enabling fallback.
    report = "/tmp/vie_model_deployment/op20_report.json"
    os.makedirs("/tmp/vie_model_deployment", exist_ok=True)
    proc = subprocess.run(
        [sys.executable, VALIDATE, "--model", MODEL, "--converted-model", MODEL,
         "--target-device", "cuda", "--build-dir", BUILD, "--report", report,
         "--skip-numerical", "--skip-performance", "--overwrite"],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    data = json.load(open(report))
    assert data["original_strict_cuda"]["status"] == "FAIL"
    assert data["original_strict_cuda"]["classification"] == \
        "cuda_ep_unsupported_op_would_fallback_to_cpu"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
