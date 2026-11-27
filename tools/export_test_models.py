"""Rebuild the Ultralytics-derived ONNX fixtures that VIE's tests use.

These weights are not distributed with the repository: they are exported from
Ultralytics checkpoints, which are AGPL-3.0. This script records the exact export
settings behind each fixture so a fresh clone can run the real-model tests.

Nothing here imports VIE. The ``.pt`` checkpoints are downloaded into a scratch
directory outside the repository and are never committed.

  python3 tools/export_test_models.py --list     # which fixtures are present
  python3 tools/export_test_models.py            # export only what is missing
  python3 tools/export_test_models.py --force    # overwrite existing fixtures
  python3 tools/export_test_models.py --only yolov8n

Two checks per fixture:

* structural (input name, input shape, opset) - the VIE recognizers key on these,
  so a mismatch is always an error;
* SHA256 - recorded with the pinned toolchain below. A different
  ultralytics/torch build serializes the graph differently, so a hash mismatch is
  a warning unless ``--strict`` is given.

Pins used for the recorded hashes: ultralytics 8.4.0, torch 2.14.1, onnx 1.23.2,
except ``yolov8n.onnx`` which was produced by a torch 2.6.0 export (opset 20).
"""

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Ultralytics-derived fixtures. ``shape`` entries are strings so symbolic dims
# ("batch") compare the same way on either side.
FIXTURES = {
    "yolov8n": dict(
        out="testdata/onnx/yolov8n.onnx",
        ckpt="yolov8n.pt",
        family="yolo",
        export=dict(format="onnx", imgsz=640, batch=1, dynamic=False, simplify=False,
                    opset=20, nms=False, device="cpu"),
        sha256="65158dad735be799c2466fa15e260c09558080bd530b42a8d0c3d1b419afd8b5",
        opset=20,
        shape=["1", "3", "640", "640"],
    ),
    "yolov8n-seg": dict(
        out="testdata/onnx/yolov8n-seg.onnx",
        ckpt="yolov8n-seg.pt",
        family="yolo",
        export=dict(format="onnx", imgsz=640, batch=1, dynamic=False, simplify=False,
                    opset=17, nms=False, device="cpu"),
        sha256="430dd97586f37e9b87940419eb54ccf7a8e6460c8218785510a714ad46c8607e",
        opset=17,
        shape=["1", "3", "640", "640"],
    ),
    "yolov8n_batch": dict(
        out="testdata/onnx_dynamic/yolov8n_batch.onnx",
        ckpt="yolov8n.pt",
        family="yolo",
        export=dict(format="onnx", imgsz=640, batch=1, dynamic=True, simplify=False,
                    opset=12, nms=False, device="cpu"),
        sha256="b272dedbaa3b36cbd73cc7c2734264119cea1870ebf6693e60663a5180028af1",
        opset=12,
        shape=["batch", "3", "640", "640"],
    ),
    "rtdetr-l": dict(
        out="testdata/onnx/rtdetr-l.onnx",
        ckpt="rtdetr-l.pt",
        family="rtdetr",
        export=dict(format="onnx", imgsz=640, batch=1, dynamic=False, simplify=False,
                    opset=17, nms=False, device="cpu"),
        sha256="91e7cb0358575e37d7d241db616af07fcce4924ca0b3598ac7e35f2ea8ff719f",
        opset=17,
        shape=["1", "3", "640", "640"],
    ),
}

# DeepLabV3 comes from torchvision (BSD-3-Clause) and stays in the repository;
# see tools/diagnostics/semantic_segmentation/export_deeplabv3.py.


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status(name: str, fixture: dict) -> str:
    path = REPO / fixture["out"]
    if not path.exists():
        return f"{name:14s} MISSING        {fixture['out']}"
    actual = sha256_of(path)
    label = "OK" if actual == fixture["sha256"] else "HASH DIFFERS"
    return f"{name:14s} {label:14s} {fixture['out']}  sha256={actual[:16]}..."


def inspect(path: Path):
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    opset = {o.domain or "ai.onnx": o.version for o in graph.opset_import}
    dims = [d.dim_param or str(d.dim_value) for d in graph.input[0].type.tensor_type.shape.dim]
    return graph.input[0].name, dims, opset.get("ai.onnx")


def build(name: str, fixture: dict, weights_dir: Path, force: bool, strict: bool) -> int:
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError as error:
        print(f"error: {error}. This script needs ultralytics, torch and onnx.", file=sys.stderr)
        return 2

    path = REPO / fixture["out"]
    if path.exists() and not force:
        print(f"{name}: {path.relative_to(REPO)} already exists, keeping it (use --force to overwrite)")
        return 0

    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = weights_dir / fixture["ckpt"]
    if not checkpoint.exists():
        print(f"{name}: Ultralytics will download {fixture['ckpt']} into {weights_dir}")

    loader = RTDETR if fixture["family"] == "rtdetr" else YOLO
    exported = Path(loader(str(checkpoint)).export(**fixture["export"])).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(exported.read_bytes())

    input_name, dims, opset = inspect(path)
    actual = sha256_of(path)
    print(f"{name}: wrote {path.relative_to(REPO)}")
    print(f"  input     {input_name} [{', '.join(dims)}]  opset={opset}")
    print(f"  sha256    {actual}")

    code = 0
    for message in (
        [] if input_name == "images" else [f"input name is {input_name!r}, expected 'images'"]
    ) + (
        [] if dims == fixture["shape"] else [f"input shape is {dims}, expected {fixture['shape']}"]
    ) + (
        [] if opset == fixture["opset"] else [f"opset is {opset}, expected {fixture['opset']}"]
    ):
        print(f"  structural mismatch: {message}", file=sys.stderr)
        code = 1
    if code:
        print("  VIE recognizers key on graph structure, so fix this before running the tests.",
              file=sys.stderr)
    elif actual != fixture["sha256"]:
        print("  hash differs: produced by a different toolchain build. Structure still matches.",
              file=sys.stderr)
        code = 1 if strict else 0
    return code


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(FIXTURES), help="handle a single fixture")
    parser.add_argument("--list", action="store_true", help="report which fixtures are present")
    parser.add_argument("--force", action="store_true", help="overwrite existing fixtures")
    parser.add_argument("--strict", action="store_true", help="treat a SHA256 mismatch as failure")
    parser.add_argument("--weights-dir", default="/tmp/vie_test_models",
                        help="scratch directory for .pt checkpoints (must stay outside the repo)")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir).resolve()
    if weights_dir == REPO or REPO in weights_dir.parents:
        print("error: --weights-dir must stay outside the repository", file=sys.stderr)
        return 2

    names = [args.only] if args.only else sorted(FIXTURES)
    if args.list:
        for name in names:
            print(status(name, FIXTURES[name]))
        print("\nDeepLabV3 and the synthetic ONNX fixtures are tracked in the repository.")
        return 0

    code = 0
    for name in names:
        code = build(name, FIXTURES[name], weights_dir, args.force, args.strict) or code
    if code == 0:
        print("\nNext: cmake --build build-ort -j && ctest --test-dir build-ort --output-on-failure")
    return code


if __name__ == "__main__":
    sys.exit(main())
