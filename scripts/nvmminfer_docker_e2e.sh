#!/usr/bin/env bash
# Dockerised end-to-end demo for the nvmminfer detector + nvmmdrawdet overlay.
# Run ON the Jetson (JetPack 6 / L4T 36.4, Orin). Builds the image, builds the
# plugins inside the container (under --runtime nvidia, so the tegra BSP libs +
# CUDA/TRT are visible), then streams an annotated H.264 feed over TCP. Watch it
# from any machine on the LAN with the printed gst-launch command.
#
# Env (override as needed):
#   ENGINE   TensorRT engine file          (default ~/yolo/yolo11n_fp16.engine)
#   VIDEO    looped H.264 elementary src   (default JetPack car sample if present)
#   IMG      still-image fallback source   (default ~/yolo/bus.jpg)
#   FPS      stream framerate              (default 60)
#   PORT     tcpserversink port            (default 6000)
#   IMAGE    docker image tag              (default gst-nvmm-infer:jp6)
#   NAME     server container name         (default nvmm-e2e)
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
# `docker -v` treats a RELATIVE host path as a named volume, not a bind-mount,
# so canonicalize every mounted path to absolute before handing it to docker.
ENGINE="$(realpath "$ENGINE")"

# A moving video makes the better demo; fall back to the still image if absent.
# Caps are single-quoted so the '(memory:NVMM)' parens survive intact when the
# pipeline string is re-parsed by the container's shell under `bash -c`.
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
# The kernel here lacks the iptables 'raw' table, so BuildKit's bridge fails —
# build with host networking.
docker build --network=host -f "$ROOT/docker/Dockerfile.jetson-jp6-infer" \
  -t "$IMAGE" "$ROOT" >/dev/null || fail "docker build failed"

echo "== [2/3] build plugins inside the container (runtime nvidia) =="
# meson setup self-skips an already-configured dir; ninja is a fast no-op when
# up to date, so this is cheap on repeat runs yet still picks up source edits.
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
