#!/bin/bash
# Corrected: both ONNX inputs are already static -> no shape flags.
set -u
T=/usr/src/tensorrt/bin/trtexec
OUT=/work/exp_ab; mkdir -p "$OUT"

echo "############ A: STATIC image_encoder (onnx is fixed 1x3x512x512) ############"
$T --onnx=/enc/image_encoder.onnx --fp16 --saveEngine=$OUT/enc_static.engine \
   > $OUT/enc_static.log 2>&1
echo "build exit=$?"
grep -iE "GPU Compute Time: .*mean|Throughput" $OUT/enc_static.log | head -2
echo "(deployed static baseline: GPU compute mean = 42.88 ms)"

echo; echo "############ B: detector on DLA core 1 (onnx fixed 1x3x576x1920) ############"
$T --onnx=/y/detector.onnx --useDLACore=1 --fp16 --allowGPUFallback \
   --saveEngine=$OUT/yolo_dla1.engine > $OUT/yolo_dla1.log 2>&1
echo "build exit=$?"
echo "DLA-run layer lines : $(grep -ic 'running on DLA' $OUT/yolo_dla1.log)"
echo "GPU-switch lines    : $(grep -ic 'Switching this layer.s device type to GPU' $OUT/yolo_dla1.log)"
echo "subgraph splits     : $(grep -ic 'Splitting DLA subgraph' $OUT/yolo_dla1.log)"
echo "16-subgraph-limit   : $(grep -ic 'only 16 subgraphs' $OUT/yolo_dla1.log)"
echo "--- unsupported ops (distinct) ---"
grep -oiE "do not support [A-Z ]+|[A-Za-z]+ is not supported on DLA" $OUT/yolo_dla1.log | sort | uniq -c | head
echo "--- DLA yolo latency @576x1920 (incl fallback+reformats) ---"
grep -iE "GPU Compute Time: .*mean|Latency: .*mean|Throughput" $OUT/yolo_dla1.log | head -3

echo; echo "############ B-ref: same detector on GPU @576x1920 (apples-to-apples) ############"
$T --onnx=/y/detector.onnx --fp16 --saveEngine=$OUT/yolo_gpu576.engine \
   > $OUT/yolo_gpu576.log 2>&1
echo "build exit=$?"
grep -iE "GPU Compute Time: .*mean|Throughput" $OUT/yolo_gpu576.log | head -2
echo "(deployed GPU yolo @1088x1920 baseline = 21.12 ms; 576x1920 is ~0.53x the pixels)"
