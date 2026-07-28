#!/usr/bin/env bash
# Per-op DLA eligibility for the three graphs the DLA verdict never measured.
# Closes the generalization it made from memory_encoder ("Hiera is attention ->
# categorically worse") with actual layer counts.
set -u
T=/usr/src/tensorrt/bin/trtexec
OUT=/o/dla_inv; mkdir -p $OUT
for G in memory_attention mask_decoder; do
  echo "############ $G on DLA core 1, fp16, GPU fallback ############"
  $T --onnx=/onnx/$G.onnx --useDLACore=1 --fp16 --allowGPUFallback \
     --verbose --skipInference > $OUT/$G.log 2>&1
  echo "build exit=$?"
  echo "  layers running on DLA : $(grep -ic "running on DLA" $OUT/$G.log)"
  echo "  layers running on GPU : $(grep -ic "running on GPU" $OUT/$G.log)"
  echo "  switched to GPU       : $(grep -ic "Switching this layer.s device type to GPU" $OUT/$G.log)"
  echo "  DLA subgraph splits   : $(grep -ic "Splitting DLA subgraph" $OUT/$G.log)"
  echo "  hit 16-subgraph limit : $(grep -ic "only 16 subgraphs" $OUT/$G.log)"
  echo "  --- distinct unsupported reasons ---"
  grep -oiE "do not support [A-Za-z ]+|[A-Za-z]+ is not supported on DLA" $OUT/$G.log | sort | uniq -c | sort -rn | head -8 | sed "s/^/    /"
done
echo DLA-INV-DONE
