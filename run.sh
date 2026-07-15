#!/usr/bin/env bash
# Zero-copy YOLO26n + SAMURAI(SAM2.1) + master-KF fusion tracker → RTP/H.264 over
# UDP (default), RTSP (mediamtx), or file. Run inside the gst-nvmm-infer:jp6 container
# (TRT 10.3 + CUDA 12.6; OpenCV-free — GMC/validity use the XFeat+LightGlue matcher,
# so with GMC=true the engine dir must also hold xfeat.engine + lightglue.engine), e.g.:
#   docker run --rm --runtime nvidia --network host \
#     -v <repo>:/src -v <workdir>:/o -v <videos>:/v gst-nvmm-infer:jp6 bash /src/run.sh
#
# Env knobs (all optional):
#   INPUT      input clip            (default: /v/drone_day1.mkv)
#   SINK       udp | rtsp | file     (default: udp)
#   DST        dest host             (default: 127.0.0.1)
#   PORT       UDP port              (default: 5600)
#   MAXKF      consecutive KF-only   (default: 2)
#   TRT        engine dir            (default: /o/trt)
#   PLUGINS    GST_PLUGIN_PATH       (default: /src/builddir-docker)
#   OUTFILE    file sink path        (default: /o/results/out.mp4)
#   SEEDROI    "x,y,w,h" force seed  (default: empty = YOLO auto-seed)
#   SEEDDELAY  frames before seeding (default: 0)
#   KFVELNOISE SAMURAI KF vel noise  (default: 0.00625 = SORT parity)
#   GMC        camera-motion comp    (default: false)
#
# v2 handheld-drone PRESET (detector is blind to this drone; seed it once, no GMC):
#   INPUT="/o/WhatsApp Video 2026-06-14 at 18.10.17.mp4" SINK=file \
#   SEEDROI="295,465,30,30" SEEDDELAY=300 KFVELNOISE=0.1 MAXKF=2 GMC=false \
#   OUTFILE=/o/results/v2_tracked_final.mp4 bash /src/run.sh
set -eu

INPUT="${INPUT:-/v/drone_day1.mkv}"
SINK="${SINK:-udp}"
DST="${DST:-127.0.0.1}"
PORT="${PORT:-5600}"
MAXKF="${MAXKF:-2}"
TRT="${TRT:-/o/trt}"
PLUGINS="${PLUGINS:-/src/builddir-docker}"
OUTFILE="${OUTFILE:-/o/results/out.mp4}"
SEEDROI="${SEEDROI:-}"
SEEDDELAY="${SEEDDELAY:-0}"
KFVELNOISE="${KFVELNOISE:-0.00625}"
GMC="${GMC:-false}"

# Optional forced seed-roi (bypass YOLO auto-seed for detector-blind targets).
SAMURAI_SEED=""
[ -n "$SEEDROI" ] && SAMURAI_SEED="seed-roi=$SEEDROI"

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
  nvmminfer engine-file="$TRT/yolo26n_1088x1920.engine" ! queue ! \
  nvmmsamurai engine-dir="$TRT" consts-file="$TRT/samurai_consts.bin" max-kf="$MAXKF" \
              seed-prefer-center=true seed-delay="$SEEDDELAY" kf-vel-noise="$KFVELNOISE" \
              gmc="$GMC" $SAMURAI_SEED ! queue ! \
  nvmmfusekf target-class=0 ! queue ! \
  nvmmdrawdet draw-det=false draw-track=true thickness=2 ! queue ! \
  nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
  nvv4l2h264enc bitrate=8000000 insert-sps-pps=1 ! h264parse ! \
  $OUT
