"""Convert an ONNX model from float16 to float32: on CPU float32 is noticeably faster.

The onnx package is not used here (on some Windows machines Smart App Control blocks its DLL) - the
protobuf is read in pure Python through the generated onnx_ml_pb2.py.

    set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    python tools/fp16_to_fp32.py models/yolo11n.onnx models/yolo11n_fp32.onnx
"""
import importlib.util
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import numpy as np  # noqa: E402

FLOAT, FLOAT16 = 1, 10


def load_pb2():
    import site
    for base in site.getsitepackages():
        path = os.path.join(base, "onnx", "onnx_ml_pb2.py")
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("onnx_ml_pb2", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("onnx/onnx_ml_pb2.py not found - run: pip install onnx")


def convert_tensor(t):
    if t.data_type != FLOAT16:
        return False
    if t.raw_data:
        arr = np.frombuffer(t.raw_data, dtype=np.float16)
    else:
        arr = np.array(t.int32_data, dtype=np.uint16).view(np.float16)
    t.ClearField("int32_data")
    t.raw_data = arr.astype(np.float32).tobytes()
    t.data_type = FLOAT
    return True


def main(src, dst):
    pb = load_pb2()
    model = pb.ModelProto()
    with open(src, "rb") as f:
        model.ParseFromString(f.read())
    g = model.graph
    n = sum(convert_tensor(t) for t in g.initializer)
    for node in g.node:
        for a in node.attribute:
            if a.HasField("t"):
                n += convert_tensor(a.t)
            for t in a.tensors:
                n += convert_tensor(t)
            if node.op_type == "Cast" and a.name == "to" and a.i == FLOAT16:
                a.i = FLOAT
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.type.tensor_type.elem_type == FLOAT16:
            vi.type.tensor_type.elem_type = FLOAT
    with open(dst, "wb") as f:
        f.write(model.SerializeToString())
    print(f"{src} -> {dst}: tensors converted: {n}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
