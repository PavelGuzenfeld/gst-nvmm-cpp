#!/usr/bin/env python3
"""Falsifier step 0 — inspect memory_encoder.onnx op graph + DLA core count.

Prints:
  - TensorRT DLA core count (confirms 2 on Orin NX 16GB)
  - op-type histogram of the ONNX
  - I/O tensor names + shapes
  - a linearized node list flagging DLA-hostile ops (LayerNorm/GELU/etc.)
    so we can see where the conv chunks are and where the cuts must land.
"""
import sys
import onnx
from collections import Counter, OrderedDict

path = sys.argv[1] if len(sys.argv) > 1 else "/onnx/memory_encoder.onnx"

# DLA core count via TRT (authoritative for the target).
try:
    import tensorrt as trt
    b = trt.Builder(trt.Logger(trt.Logger.ERROR))
    print(f"[trt] version={trt.__version__} num_DLA_cores={b.num_DLA_cores}")
except Exception as e:
    print(f"[trt] could not query DLA cores: {e}")

m = onnx.load(path)
g = m.graph
print(f"\n[onnx] ir_version={m.ir_version} producer={m.producer_name} "
      f"opset={ {o.domain or 'ai.onnx': o.version for o in m.opset_import} }")

def shp(vi):
    d = vi.type.tensor_type.shape.dim
    return [x.dim_value if x.HasField('dim_value') else (x.dim_param or '?') for x in d]

print("\n[inputs]")
for i in g.input:
    print(f"  {i.name}: {shp(i)}")
print("[outputs]")
for o in g.output:
    print(f"  {o.name}: {shp(o)}")

hist = Counter(n.op_type for n in g.node)
print(f"\n[op histogram] {len(g.node)} nodes")
for op, c in hist.most_common():
    print(f"  {op:24s} {c}")

# DLA-hostile op types (TRT DLA can't take these; they force GPU / cut points).
HOSTILE = {"LayerNormalization", "InstanceNormalization", "Gelu", "Erf",
           "ReduceMean", "Softmax", "Pow", "Sqrt", "Div", "Sub", "Where",
           "Cast", "Tanh", "Reciprocal"}
print("\n[linear node walk — * = DLA-hostile / likely cut boundary]")
for idx, n in enumerate(g.node):
    flag = " *" if n.op_type in HOSTILE else ""
    if flag or n.op_type in ("Conv", "Resize", "Add", "Mul", "Concat"):
        print(f"  {idx:3d} {n.op_type:22s}{flag}")
