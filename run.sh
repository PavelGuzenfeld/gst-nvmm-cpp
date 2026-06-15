#!/usr/bin/env bash
# Zero-copy YOLO26n + SAMURAI(SAM2.1) + master-KF fusion tracker → RTP/H.264 over
# UDP (default), RTSP (mediamtx), or file. Run inside the gst-nvmm-infer:jp6
# container (TRT 10.3 engines), e.g.:
#   docker run --rm --runtime nvidia --network host \
#     -v <repo>:/src -v <workdir>:/o -v <videos>:/v gst-nvmm-infer:jp6 bash /src/run.sh
#
# Env knobs (all optional):
#   INPUT   input clip            (default: /v/clip.mp4)
#   SINK    udp | rtsp | file     (default: udp)
#   DST     dest host             (default: 127.0.0.1)
#   PORT    UDP port              (default: 5600)
#   MAXKF   consecutive KF-only   (default: 2)
#   TRT     engine dir            (default: /o/trt)
#   PLUGINS GST_PLUGIN_PATH       (default: /src/builddir-docker)
#   OUTFILE file sink path        (default: /o/results/out.mp4)
set -eu

INPUT="${INPUT:-/v/clip.mp4}"
SINK="${SINK:-udp}"
DST="${DST:-127.0.0.1}"
PORT="${PORT:-5600}"
MAXKF="${MAXKF:-2}"
TRT="${TRT:-/o/trt}"
PLUGINS="${PLUGINS:-/src/builddir-docker}"
OUTFILE="${OUTFILE:-/o/results/out.mp4}"

case "$INPUT" in
  *.mkv) DEMUX=matroskademux ;;
  *)     DEMUX=qtdemux ;;
esac

case "$SINK" in
  udp)  OUT="rtph264pay config-interval=1 pt=96 ! udpsink host=$DST port=$PORT" ;;
  rtsp) OUT="rtspclientsink location=rtsp://$DST:8554/track" ;;       # mediamtx
  file) mkdir -p "$(dirname "$OUTFILE")"; OUT="qtmux ! filesink location=$OUTFILE" ;;
  *)    echo "SINK must be udp|rtsp|file" >&2; exit 2 ;;
esac

echo "tracker: INPUT=$INPUT SINK=$SINK DST=$DST:$PORT MAXKF=$MAXKF"
export GST_PLUGIN_PATH="$PLUGINS"
exec gst-launch-1.0 -e \
  filesrc location="$INPUT" ! "$DEMUX" ! h264parse ! nvv4l2decoder ! queue ! \
  nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! queue ! \
  nvmminfer engine-file="$TRT/yolo.engine" ! queue ! \
  nvmmsamurai engine-dir="$TRT" consts-file="$TRT/samurai_consts.bin" max-kf="$MAXKF" seed-prefer-center=true ! queue ! \
  nvmmfusekf target-class=0 ! queue ! \
  nvmmdrawdet draw-det=false thickness=2 ! queue ! \
  nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
  nvv4l2h264enc bitrate=8000000 insert-sps-pps=1 ! h264parse ! \
  $OUT
