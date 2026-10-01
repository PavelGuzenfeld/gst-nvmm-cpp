#!/usr/bin/env python3
"""Print TRT's DLA core count (2 on Orin NX 16GB), then an ONNX graph's I/O, op histogram and a
node walk that flags DLA-hostile ops, which force GPU fallback and mark where cuts must land."""
import sys
import onnx
from collections import Counter, OrderedDict

path = sys.argv[1] if len(sys.argv) > 1 else "/onnx/memory_encoder.onnx"

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

DLA_HOSTILE_OPS = {"LayerNormalization", "InstanceNormalization", "Gelu", "Erf",
           "ReduceMean", "Softmax", "Pow", "Sqrt", "Div", "Sub", "Where",
           "Cast", "Tanh", "Reciprocal"}
print("\n[linear node walk — * = DLA-hostile / likely cut boundary]")
for idx, n in enumerate(g.node):
    flag = " *" if n.op_type in DLA_HOSTILE_OPS else ""
    if flag or n.op_type in ("Conv", "Resize", "Add", "Mul", "Concat"):
        print(f"  {idx:3d} {n.op_type:22s}{flag}")
