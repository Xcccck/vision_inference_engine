#!/usr/bin/env python3
"""Phase 9.3.3 (Resize CUDA-EP compatibility investigation).

Read-only analyzer for the two YOLOv8n `Resize` nodes: it dumps every Resize
node's attributes, inputs (X / roi / scales / sizes), whether each non-X input
is a graph constant/initializer or produced dynamically by other nodes, the
shapes/dtypes from ONNX shape inference, and the producer/consumer chain.

It does NOT modify the model. It only requires `onnx` (available in this env).
The CUDA-EP placement conclusion is NOT made here -- this file gives the
static-graph facts (Phase 2/3) that the ORT CUDA experiments are compared
against.

Usage:
    python3 inspect_resize_nodes.py [--model PATH]
"""
import argparse
import hashlib
import sys

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError:  # pragma: no cover - env guard
    sys.stderr.write("error: python package 'onnx' is required\n")
    sys.exit(1)

DIM_TYPE = {0: "DIMENSION", 1: "FLOAT", 2: "INT"}


def dtype_name(elem):
    return TensorProto.DataType.Name(elem) if elem in TensorProto.DataType.keys() else str(elem)


def shape_of(value_info):
    t = value_info.type.tensor_type
    dims = []
    for d in t.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        elif d.HasField("dim_param"):
            dims.append(f"{d.dim_param}?")
        else:
            dims.append("?")
    return f"[{', '.join(str(x) for x in dims)}] {dtype_name(t.elem_type)}"


def initializer_map(model):
    return {init.name: init for init in model.graph.initializer}


def node_outputs(model):
    prod = {}
    for n in model.graph.node:
        for o in n.output:
            prod[o] = n
    return prod


def value_infos(model):
    m = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        m[vi.name] = vi
    return m


def describe_resize(node, model, inits, vi_map, producers):
    print(f"\n=== Resize node: {node.name!r} (opset domain={node.domain!r}) ===")
    print(f"  Operator: {node.op_type}")
    print("  Attributes (ONLY those actually present in the file):")
    for a in node.attribute:
        if a.type == a.INT:
            print(f"    {a.name} = {a.i}  ({DIM_TYPE.get(a.i, '?')})")
        elif a.type == a.FLOAT:
            print(f"    {a.name} = {a.f}")
        elif a.type == a.STRING:
            print(f"    {a.name} = {a.s.decode()}")
        elif a.type == a.INTS:
            print(f"    {a.name} = {list(a.ints)}")
        elif a.type == a.FLOATS:
            print(f"    {a.name} = {list(a.floats)}")
        else:
            print(f"    {a.name} = <type {a.type}>")
    print("  Inputs:")
    for idx, name in enumerate(node.input):
        roles = ["X", "roi", "scales", "sizes"]
        role = roles[idx] if idx < len(roles) else f"#{idx}"
        if name == "":
            print(f"    [{role}] <omitted empty string input '{name}'>")
            continue
        if name in inits:
            init = inits[name]
            try:
                from onnx import numpy_helper
                arr = numpy_helper.to_array(init)
                vals = arr.flatten().tolist()[:8]
            except Exception as e:  # noqa
                vals = f"<unreadable: {e}>"
            print(f"    [{role}] '{name}' : INITIALIZER shape={list(init.dims)} dtype={dtype_name(init.data_type)} values={vals}")
        elif name in producers:
            p = producers[name]
            print(f"    [{role}] '{name}' : DYNAMIC, produced by node '{p.name}' op={p.op_type}")
        else:
            src = "GRAPH-INPUT" if any(i.name == name for i in model.graph.input) else "UNKNOWN"
            print(f"    [{role}] '{name}' : {src} (no initializer, no in-graph producer)")
        if name in vi_map:
            print(f"           shape/dtype: {shape_of(vi_map[name])}")
    print("  Outputs:")
    for name in node.output:
        s = shape_of(vi_map[name]) if name in vi_map else "unknown"
        print(f"    '{name}' : {s}")
    cons = [n.name + "(" + n.op_type + ")" for n in model.graph.node if name_present(node.output, n.input)]
    print(f"  Consumer nodes: {cons if cons else 'none'}")


def name_present(outputs, inputs):
    return any(o in inputs for o in outputs)


def producer_of(node, producers):
    out = []
    for i in node.input:
        if i in producers:
            out.append(producers[i].name + "(" + producers[i].op_type + ")")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="testdata/onnx/yolov8n.onnx")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        raw = f.read()
    print(f"Model: {args.model}")
    print(f"  size_bytes: {len(raw)}")
    print(f"  sha256: {hashlib.sha256(raw).hexdigest()}")

    model = onnx.load(args.model)
    print(f"  IR version: {model.ir_version}")
    print(f"  producer: {model.producer_name!r} {model.producer_version!r}")
    print(f"  opset_import: " + ", ".join(f"{op.domain or 'ai.onnx'}={op.version}" for op in model.opset_import))
    print(f"  graph name: {model.graph.name!r}")
    print(f"  node count: {len(model.graph.node)}")
    op_counts = {}
    for n in model.graph.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
    print(f"  op_type histogram (top): " + ", ".join(f"{k}={v}" for k, v in sorted(op_counts.items(), key=lambda x: -x[1])))
    print(f"  Resize count: {op_counts.get('Resize', 0)}")

    inits = initializer_map(model)
    producers = node_outputs(model)
    vi_map = value_infos(model)

    # ONNX checker + shape inference (separate concerns from CUDA support).
    print("\n=== ONNX checker ===")
    try:
        onnx.checker.check_model(model)
        print("  check_model: PASS")
    except Exception as e:  # noqa
        print(f"  check_model: FAIL -> {e}")

    print("=== ONNX shape inference ===")
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        vi_map = value_infos(inferred)
        print(f"  infer_shapes: OK (value_info count {len(inferred.graph.value_info)})")
        strict = None
        try:
            onnx.shape_inference.infer_shapes(model, strict_mode=True)
            strict = "PASS"
        except Exception as e:  # noqa
            strict = f"FAIL -> {e}"
        print(f"  infer_shapes(strict_mode=True): {strict}")
    except Exception as e:  # noqa
        print(f"  infer_shapes: FAIL -> {e}")

    print("\n=== All Resize nodes ===")
    resize_nodes = [n for n in model.graph.node if n.op_type == "Resize"]
    for n in resize_nodes:
        describe_resize(n, model, inits, vi_map, producers)
        print(f"  Producer nodes (of inputs): {producer_of(n, producers)}")

    print("\nNOTE: onnx.checker / shape-inference passing does NOT imply CUDA EP")
    print("      executes the node. Placement is decided by ORT (see C++ experiments).")


if __name__ == "__main__":
    main()
