#!/usr/bin/env python3
"""Phase 9.3.3: build minimal ONNX Resize models that reproduce YOLOv8n's two
Resize nodes exactly, plus controlled variants, for CUDA-EP placement probing.

Nothing here touches VIE or the baseline model; outputs go to --outdir (default
/tmp/resize_min). The "yolov8_exact" variant copies the real attributes read
from the baseline graph:
    mode=nearest, coordinate_transformation_mode=asymmetric, nearest_mode=floor,
    cubic_coeff_a=-0.75, scales=[1,1,2,2] constant, static 4D NCHW fp32.
Other variants change ONE factor at a time (coordinate mode, interpolation mode,
sizes-vs-scales, dynamic scales, rank, dynamic batch) so the C++ probe can
pinpoint which constraint pushes Resize to the CPU EP.

Usage: python3 build_minimal_resize_model.py --outdir /tmp/resize_min
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 20


def finish(graph_name, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    model.producer_name = "resize_cuda_compat_probe"
    onnx.checker.check_model(model)
    return model


def scales_const(values):
    return numpy_helper.from_array(np.array(values, dtype=np.float32), "sc")


def one_resize(node_attrs, in_dims, out_dims, input_extra=None, initializer=None):
    """Single Resize with X as graph input; optional extra input/initializer."""
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, in_dims)]
    node_inputs = ["X", ""]  # X, roi(omitted)
    inits = []
    if initializer is not None:
        node_inputs.append(initializer.name)
        inits.append(initializer)
    else:
        node_inputs.append("")
    if input_extra:
        inputs.extend(input_extra["graph_inputs"])
        node_inputs = node_inputs[:2] + input_extra["resize_inputs"]
    node = helper.make_node("Resize", node_inputs, ["Y"], name="Resize", **node_attrs)
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_dims)]
    return finish("resize_min", [node], inputs, outputs, inits)


def write(model, outdir, fname):
    path = os.path.join(outdir, fname)
    onnx.save(model, path)
    print(f"  wrote {os.path.basename(path)} (nodes={len(model.graph.node)}, "
          f"in={list(model.graph.input[0].type.tensor_type.shape.dim)} )")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/resize_min")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    yolov8 = dict(mode="nearest", coordinate_transformation_mode="asymmetric",
                  nearest_mode="floor", cubic_coeff_a=-0.75)
    sc = scales_const([1.0, 1.0, 2.0, 2.0])

    print("Building minimal Resize models:")
    # 1) exact baseline reproduction.
    write(one_resize(yolov8, [1, 256, 20, 20], [1, 256, 40, 40], initializer=sc),
          args.outdir, "resize_yolov8_exact.onnx")
    # 2) only coordinate_transformation_mode -> half_pixel.
    a2 = dict(yolov8, coordinate_transformation_mode="half_pixel")
    write(one_resize(a2, [1, 256, 20, 20], [1, 256, 40, 40], initializer=scales_const([1.0, 1.0, 2.0, 2.0])),
          args.outdir, "resize_halfpixel.onnx")
    # 3) only mode -> linear.
    a3 = dict(mode="linear", coordinate_transformation_mode="asymmetric", cubic_coeff_a=-0.75)
    write(one_resize(a3, [1, 256, 20, 20], [1, 256, 40, 40], initializer=sc),
          args.outdir, "resize_linear.onnx")
    # 4) sizes input instead of scales (sizes must be int64 per ONNX spec).
    a4 = dict(mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    sz = numpy_helper.from_array(np.array([1, 256, 40, 40], dtype=np.int64), "sz")
    node4 = helper.make_node("Resize", ["X", "", "", "sz"], ["Y"], name="Resize", **a4)
    graph4 = helper.make_graph([node4], "resize_min",
                               [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 256, 20, 20])],
                               [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 256, 40, 40])],
                               initializer=[sz])
    m4 = helper.make_model(graph4, opset_imports=[helper.make_opsetid("", OPSET)]); m4.ir_version = 9
    onnx.checker.check_model(m4)
    write(m4, args.outdir, "resize_sizes.onnx")
    # 5) dynamic scales (scales is a graph input, not constant).
    node5 = helper.make_node("Resize", ["X", "", "scales_in"], ["Y"], name="Resize", **yolov8)
    graph5 = helper.make_graph([node5], "resize_min",
                               [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 256, 20, 20]),
                                helper.make_tensor_value_info("scales_in", TensorProto.FLOAT, [4])],
                               [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["b", "c", "h", "w"])])
    m5 = helper.make_model(graph5, opset_imports=[helper.make_opsetid("", OPSET)]); m5.ir_version = 9
    onnx.checker.check_model(m5)
    write(m5, args.outdir, "resize_dynamic_scales.onnx")
    # 6) 3D input (rank-3).
    a6 = dict(mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor", cubic_coeff_a=-0.75)
    write(one_resize(a6, [1, 64, 20], [1, 64, 40], initializer=scales_const([1.0, 1.0, 2.0])),
          args.outdir, "resize_3d.onnx")
    # 7) dynamic batch (dim 0 symbolic).
    node7 = helper.make_node("Resize", ["X", "", "sc"], ["Y"], name="Resize", **yolov8)
    graph7 = helper.make_graph([node7], "resize_min",
                               [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 256, 20, 20])],
                               [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 256, 40, 40])],
                               initializer=[scales_const([1.0, 1.0, 2.0, 2.0])])
    m7 = helper.make_model(graph7, opset_imports=[helper.make_opsetid("", OPSET)]); m7.ir_version = 9
    onnx.checker.check_model(m7)
    write(m7, args.outdir, "resize_dynamic_batch.onnx")
    print("done")


if __name__ == "__main__":
    main()
