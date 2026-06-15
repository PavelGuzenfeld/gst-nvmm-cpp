# Single-object tracker pipeline

A zero-copy GStreamer tracker on Jetson Orin NX: a **YOLO** detector →
**SAMURAI (SAM 2.1)** visual tracker → **master Kalman fusion** → overlay →
RTP/H.264 (or file). Everything runs inside the `gst-nvmm-infer:jp6` container
(TensorRT 10.3, CUDA 12.6).

The pipeline tracks a single target of a chosen COCO class (`target-class`,
default `0`) — nothing in it is application-specific.

---

## 0. Prerequisites

- Jetson Orin NX, JetPack 6.x (L4T r36).
- Docker image `gst-nvmm-infer:jp6` — build it from
  `docker/Dockerfile.jetson-jp6-infer`:
  ```bash
  docker build --network=host -f docker/Dockerfile.jetson-jp6-infer -t gst-nvmm-infer:jp6 .
  ```
- Always run the container with NVIDIA runtime + host networking:
  `--runtime nvidia --network host`.
- Engines + constants in an engine dir (`/o/trt` below):
  - `yolo.engine` (any Ultralytics YOLO detector exported to TensorRT)
  - SAMURAI: `image_encoder_bplus_512.engine`, `prompt_encoder.engine`,
    `mask_decoder.engine`, `memory_encoder.engine`, `memory_attention.engine`
  - `samurai_consts.bin` (learned out-of-engine constants)

  These are built entirely from public weights (Ultralytics YOLO + Meta SAM 2.1
  `base_plus`) — see **Building engines** (`tools/samurai/`) for the export →
  build → pack chain.
- A test clip. Any `.mp4`/`.mkv` works; the SAM 2 sample videos or an
  Ultralytics sample asset are convenient public inputs.

Mounts used throughout: repo at `/src`, working dir (engines, clips, results)
at `/o`, videos at `/v`.

---

## 1. Build

```bash
cd <repo>                       # the gst-nvmm-cpp checkout
docker run --rm --runtime nvidia --network host \
  -v "$PWD":/src -w /src gst-nvmm-infer:jp6 bash -c '
    meson setup builddir-docker        # first time only
    ninja -C builddir-docker
'
```

Re-running `ninja` after edits is enough. If you change a **build option** or
`meson_options.txt`, add `meson setup --reconfigure builddir-docker`.

The plugins land in `builddir-docker/gst/*/`; point GStreamer at them with
`GST_PLUGIN_PATH=builddir-docker`. See
[Getting started](getting-started.md#build-time-options-cuda-targets) for the
CUDA build options (`gpu_arch`, `gpu_cxx_std`).

---

## 2. Run

The `run.sh` launcher wraps the whole pipeline:

```bash
docker run --rm --runtime nvidia --network host \
  -v <repo>:/src -v <workdir>:/o -v <videos>:/v \
  gst-nvmm-infer:jp6 bash /src/run.sh
```

All knobs are environment variables (defaults shown):

| Var | Default | Meaning |
|-----|---------|---------|
| `INPUT` | `/v/clip.mp4` | input clip (`.mkv`→matroskademux, else qtdemux) |
| `SINK` | `udp` | `udp` \| `rtsp` \| `file` |
| `DST` | `127.0.0.1` | destination host (udp/rtsp) |
| `PORT` | `5600` | UDP port |
| `MAXKF` | `2` | consecutive Kalman-only frames between full inferences |
| `TRT` | `/o/trt` | engine directory |
| `PLUGINS` | `/src/builddir-docker` | `GST_PLUGIN_PATH` |
| `OUTFILE` | `/o/results/out.mp4` | output path when `SINK=file` |

Examples:

```bash
# stream H.264/RTP over UDP to a viewer at 192.168.1.10:5600
INPUT=/v/clip.mp4 SINK=udp DST=192.168.1.10 bash /src/run.sh

# write an annotated .mp4
INPUT=/v/clip.mp4 SINK=file OUTFILE=/o/results/clip_tracked.mp4 bash /src/run.sh
```

Receive the UDP stream on the viewer machine:

```bash
gst-launch-1.0 udpsrc port=5600 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" \
  ! rtph264depay ! h264parse ! avdec_h264 ! autovideosink sync=false
```

---

## 3. The pipeline

```
filesrc → demux → h264parse → nvv4l2decoder → queue
  → nvvidconv → video/x-raw(memory:NVMM),NV12 → queue
  → nvmminfer      (YOLO: detections → GstNvmmDetMeta)
  → nvmmsamurai    (SAM2.1 tracker: track box → GstNvmmTrackMeta)
  → nvmmfusekf     (master KF: fuse SAMURAI + YOLO, reseed authority)
  → nvmmdrawdet    (overlay track box + live FPS/coverage HUD)
  → nvvidconv → NV12 → nvv4l2h264enc → h264parse → {udpsink | rtspclientsink | filesink}
```

- **[`nvmminfer`](elements/nvmminfer.md)** runs YOLO every frame, emits detections.
- **[`nvmmsamurai`](elements/nvmmsamurai.md)** runs the SAMURAI tracker (5 TRT
  engines, on-device memory ring). Full inference on most frames; `max-kf`
  frames in between are Kalman-only for throughput. Seeds from YOLO (or a forced
  ROI).
- **[`nvmmfusekf`](elements/nvmmfusekf.md)** is the single source of truth:
  SAMURAI drives it, YOLO refines it (distance-gated). On loss it emits an
  upstream `nvmm-reseed` event back to `nvmmsamurai`.
- **[`nvmmdrawdet`](elements/nvmmdrawdet.md)** draws the fused track box + a live
  FPS / coverage HUD.

`queue` elements between stages give pipeline parallelism (each stage on its own
thread) — a real throughput win, keep them.

---

## 4. Tuning

Every element exposes range-checked GObject properties — inspect any with
`gst-inspect-1.0 <element>`. The full property tables live on the element pages:
[`nvmmsamurai`](elements/nvmmsamurai.md),
[`nvmmfusekf`](elements/nvmmfusekf.md),
[`nvmmdrawdet`](elements/nvmmdrawdet.md).

A normal run is silent. Opt into detail per element with, e.g.,
`GST_DEBUG=nvmmsamurai:6,nvmmfusekf:6` (per-frame) or `:5` (state changes only).

---

## Notes

- Build & run **only** through Docker on the Jetson.
- `nvmminfer` / `nvmmsamurai` are Jetson-only (TRT + CUDA); the pure-host
  elements (`nvmmfusekf`, etc.) build everywhere, including x86 CI.
