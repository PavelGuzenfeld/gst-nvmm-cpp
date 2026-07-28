#!/bin/bash
# Falsifier — whole memory_encoder.onnx: DLA layer placement + GPU baseline.
# Runs inside the jp6 container. ONNX at /onnx, outputs to /work.
set -u
TRTEXEC=/usr/src/tensorrt/bin/trtexec
ONNX=/onnx/memory_encoder.onnx
OUT=/work/falsify_out
mkdir -p "$OUT"

echo "############ DLA core count ############"
python3 -c 'import tensorrt as trt; b=trt.Builder(trt.Logger(trt.Logger.ERROR)); print("num_DLA_cores =", b.num_DLA_cores); print("trt =", trt.__version__)'

echo; echo "############ PASS 1: DLA core 0, fp16, GPU fallback (layer placement) ############"
$TRTEXEC --onnx=$ONNX --useDLACore=0 --fp16 --allowGPUFallback \
  --saveEngine=$OUT/memenc_dla_fb.engine --verbose \
  > $OUT/dla_fallback.log 2>&1
echo "exit=$? (build)"
echo "--- layers assigned to DLA ---"
grep -iE "running on DLA|\[DLA\]|DLA node|on DLA" $OUT/dla_fallback.log | head -40
echo "--- device-assignment summary (Reformat/DLA/GPU) ---"
grep -icE "running on DLA" $OUT/dla_fallback.log | sed 's/^/DLA layer lines: /'
grep -icE "running on GPU" $OUT/dla_fallback.log | sed 's/^/GPU layer lines: /'
grep -icE "Reformat" $OUT/dla_fallback.log | sed 's/^/Reformat lines:  /'
echo "--- number of DLA subgraphs / loadables ---"
grep -iE "DLA Node|subgraph|Number of DLA" $OUT/dla_fallback.log | head
echo "--- fallback perf (end-to-end) ---"
grep -iE "Throughput|mean =|GPU Compute Time: .*mean" $OUT/dla_fallback.log | head

echo; echo "############ PASS 2: GPU-only fp16 baseline (kill-criterion denominator) ############"
$TRTEXEC --onnx=$ONNX --fp16 \
  --saveEngine=$OUT/memenc_gpu.engine \
  > $OUT/gpu_baseline.log 2>&1
echo "exit=$? (build)"
grep -iE "Throughput|GPU Compute Time: .*mean|Latency: .*mean" $OUT/gpu_baseline.log | head

echo; echo "############ PASS 3: whole graph forced DLA, NO fallback (does it compile standalone?) ############"
$TRTEXEC --onnx=$ONNX --useDLACore=0 --fp16 \
  --saveEngine=$OUT/memenc_dla_only.engine \
  > $OUT/dla_only.log 2>&1
echo "exit=$? (build)  # nonzero => whole engine cannot be one DLA loadable (expected)"
grep -iE "not supported|falling back|no implementation|Internal Error|failed|DLA" $OUT/dla_only.log | grep -iv "loaded\|deserial" | head -20
