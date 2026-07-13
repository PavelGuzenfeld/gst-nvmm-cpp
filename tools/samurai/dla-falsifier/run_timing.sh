#!/bin/bash
# Run the instrumented nvmmsamurai on a clip, full inference every frame,
# capture per-frame SAMURAI_TIMING lines to /work/timing.log.
set -u
D=/home/nvidia/personalspace/gst-nvmm-cpp-gmc
export GST_PLUGIN_PATH=$D/builddir
export LD_LIBRARY_PATH=$D/builddir/gst/common:${LD_LIBRARY_PATH:-}
export SAMURAI_TIMING=1
export GST_DEBUG=nvmmsamurai:3

echo "=== inspect: confirm instrumented plugin loads + props ==="
gst-inspect-1.0 nvmmsamurai >/dev/null 2>&1 && echo "nvmmsamurai OK" || { echo "PLUGIN LOAD FAILED"; gst-inspect-1.0 nvmmsamurai 2>&1 | tail -5; exit 1; }

echo "=== run 250 frames, max-kf=0, seed-roi center ==="
gst-launch-1.0 -e \
  filesrc location=/o/wa1.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! queue ! \
  nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! queue ! \
  nvmmsamurai engine-dir=/o/trt consts-file=/o/trt/samurai_consts.bin \
              max-kf=0 seed-roi="900,480,120,120" seed-delay=0 ! \
  fakesink num-buffers=250 sync=false \
  2> >(grep --line-buffered "SAMURAI_TIMING" > /work/timing.log)
echo "=== captured $(wc -l < /work/timing.log) timing lines ==="
