#!/usr/bin/env bash
# JP6 Orin: build the infer image and plugins (--runtime nvidia), stream nvmminfer + nvmmdrawdet H.264 over TCP.
# Mounted paths go absolute (docker -v reads relative ones as volumes); caps stay quoted for the container's bash -c.
# Builds use host networking: this kernel lacks the iptables 'raw' table BuildKit's bridge needs.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE="${ENGINE:-$HOME/yolo/yolo11n_fp16.engine}"
VIDEO="${VIDEO:-/usr/src/jetson_multimedia_api/data/Video/sample_outdoor_car_1080p_10fps.h264}"
IMG="${IMG:-$HOME/yolo/bus.jpg}"
FPS="${FPS:-60}"
PORT="${PORT:-6000}"
IMAGE="${IMAGE:-gst-nvmm-infer:jp6}"
NAME="${NAME:-nvmm-e2e}"

fail() { echo "E2E FAIL: $1" >&2; exit 1; }
case "$FPS"  in ''|*[!0-9]*) fail "FPS must be a positive integer (got: $FPS)";; esac
case "$PORT" in ''|*[!0-9]*) fail "PORT must be a positive integer (got: $PORT)";; esac
[ -f "$ENGINE" ] || fail "engine not found: $ENGINE (build with trtexec)"
ENGINE="$(realpath "$ENGINE")"

if [ -f "$VIDEO" ]; then
  VIDEO="$(realpath "$VIDEO")"
  SOURCE="multifilesrc location=/data/src.h264 loop=true caps='video/x-h264,framerate=$FPS/1' \
    ! h264parse ! nvv4l2decoder ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12'"
  SRC_MOUNT=(-v "$VIDEO":/data/src.h264:ro)
  echo "source: looped video $VIDEO @ ${FPS}fps"
else
  [ -f "$IMG" ] || fail "neither VIDEO ($VIDEO) nor IMG ($IMG) found"
  IMG="$(realpath "$IMG")"
  SOURCE="filesrc location=/data/src.jpg ! jpegdec ! videoconvert ! imagefreeze \
    ! 'video/x-raw,format=NV12,framerate=$FPS/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12'"
  SRC_MOUNT=(-v "$IMG":/data/src.jpg:ro)
  echo "source: still image $IMG @ ${FPS}fps (no JetPack video sample found)"
fi

GST_PLUGIN_PATH=/src/builddir-docker/gst/nvmminfer:/src/builddir-docker/gst/nvmmdrawdet:/src/builddir-docker/gst/nvmmalloc

echo "== [1/3] build image $IMAGE =="
docker build --network=host -f "$ROOT/docker/Dockerfile.jetson-jp6-infer" \
  -t "$IMAGE" "$ROOT" >/dev/null || fail "docker build failed"

echo "== [2/3] build plugins inside the container (runtime nvidia) =="
docker run --rm --runtime nvidia --network host -v "$ROOT":/src -w /src "$IMAGE" \
  bash -c 'meson setup builddir-docker -Dbuildtype=debugoptimized -Dwerror=false || true
           ninja -C builddir-docker' || fail "in-container build failed"

echo "== [3/3] launch streaming server container '$NAME' on port $PORT =="
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --runtime nvidia --network host \
  -v "$ROOT":/src -v "$ENGINE":/data/engine:ro "${SRC_MOUNT[@]}" -w /src \
  -e GST_PLUGIN_PATH="$GST_PLUGIN_PATH" "$IMAGE" \
  bash -c "gst-launch-1.0 -e $SOURCE \
    ! nvmminfer engine-file=/data/engine \
    ! nvmmdrawdet ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 \
    ! matroskamux streamable=true ! tcpserversink host=0.0.0.0 port=$PORT" >/dev/null \
  || fail "server container failed to start"

sleep 8
docker ps --filter "name=$NAME" --format '{{.Names}} {{.Status}}' | grep -q Up \
  || { docker logs "$NAME" 2>&1 | tail -20; fail "server not running"; }

IP="$(hostname -I | awk '{print $1}')"
echo
echo "E2E server up: $NAME streaming on ${IP:-<jetson-ip>}:$PORT"
echo "Watch live from any machine on the LAN:"
echo "  gst-launch-1.0 tcpclientsrc host=${IP:-<jetson-ip>} port=$PORT \\"
echo "    ! matroskademux ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false"
echo "Stop:  docker rm -f $NAME"
