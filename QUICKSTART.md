# Quick start — drone tracker pipeline

Zero-copy GStreamer tracker on Jetson Orin NX: **YOLO26n** detector →
**SAMURAI (SAM2.1)** visual tracker → **master Kalman fusion** → overlay →
RTP/H.264 (or file). Everything runs inside the `gst-nvmm-infer:jp6` container
(TensorRT 10.3, CUDA 12.6).

---

## 0. Prerequisites

- Jetson Orin NX, JetPack 6.x (L4T r36).
- Docker image `gst-nvmm-infer:jp6` (TRT 10.3 + CUDA 12.6 + GStreamer).
- Always run the container with NVIDIA runtime + host networking:
  `--runtime nvidia --network host` (bridge networking is broken on this setup).
- Engines + constants in an engine dir (`/o/trt` below):
  - `yolo26n_1088x1920.engine`
  - SAMURAI: `image_encoder_bplus_512.engine`, `prompt_encoder.engine`,
    `mask_decoder.engine`, `memory_encoder.engine`, `memory_attention.engine`
  - `samurai_consts.bin` (learned out-of-engine constants)

Mounts used throughout: repo at `/src`, working dir (engines, clips, results)
at `/o`.

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

Re-running `ninja` after edits is enough. If you change a **build option** (see
below) or `meson_options.txt`, add `meson setup --reconfigure builddir-docker`.

The GStreamer plugins land in `builddir-docker/gst/*/` — point GStreamer at them
with `GST_PLUGIN_PATH=builddir-docker` (the `:` -separated subdirs, or just the
top builddir which GStreamer scans recursively).

### Build-time options (CUDA targets)

| Option | Values | Default | Meaning |
|--------|--------|---------|---------|
| `gpu_arch` | `sm_72…sm_90` | `sm_87` | GPU arch (`sm_87` = Orin, `sm_72` = Xavier) |
| `gpu_cxx_std` | `c++17`, `c++20` | `c++20` | C++ standard for the SAMURAI kernels |

Set at configure time: `meson setup builddir-docker -Dgpu_arch=sm_87 -Dgpu_cxx_std=c++20`
(or `meson configure builddir-docker -Dgpu_arch=...` then reconfigure).

---

## 2. Run

The `run.sh` launcher wraps the whole pipeline. Run it inside the container:

```bash
docker run --rm --runtime nvidia --network host \
  -v <repo>:/src -v <workdir>:/o -v <videos>:/v \
  gst-nvmm-infer:jp6 bash /src/run.sh
```

All knobs are environment variables (defaults shown):

| Var | Default | Meaning |
|-----|---------|---------|
| `INPUT` | `/v/drone_day1.mkv` | input clip (`.mkv`→matroskademux, else qtdemux) |
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
INPUT=/v/clip.mp4 SINK=udp DST=192.168.1.10 run.sh

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
  → nvmminfer      (YOLO26n: detections → GstNvmmDetMeta)
  → nvmmsamurai    (SAM2.1 tracker: track box → GstNvmmTrackMeta)
  → nvmmfusekf     (master KF: fuse SAMURAI + YOLO, reseed authority)
  → nvmmdrawdet    (overlay track box + live FPS/coverage HUD)
  → nvvidconv → NV12 → nvv4l2h264enc → h264parse → {udpsink | rtspclientsink | filesink}
```

- **nvmminfer** runs YOLO every frame, emits detections.
- **nvmmsamurai** runs the SAMURAI tracker (5 TRT engines, on-device memory
  ring). Full inference on most frames; `max-kf` frames in between are
  Kalman-only (no engines) for throughput. Seeds from YOLO (or a forced ROI).
- **nvmmfusekf** is the single source of truth: SAMURAI drives it (primary),
  YOLO refines it (secondary, distance-gated). On loss it emits an upstream
  `nvmm-reseed` event back to `nvmmsamurai`.
- **nvmmdrawdet** draws the fused track box + a live FPS / coverage HUD.

`queue` elements between stages give pipeline parallelism (each stage on its own
thread) — a real throughput win, keep them.

---

## 4. Tunable element properties

All are validated (range-checked) GObject properties — inspect any element with
`gst-inspect-1.0 <element>`.

**nvmmsamurai**
- `engine-dir`, `consts-file` — paths (required)
- `max-kf` (0–30, def 2) — Kalman-only frames between full inferences
- `kf-score-weight` (0–1, def 0.25) — stable-regime weight: `w·kf_iou + (1−w)·iou`
- `iou-threshold` (0–1, def 0.5) — min IoU to accept a Kalman update
- `kf-min-area` (def 25) — min KF box area (px²) to accept an update
- `stable-frames-threshold` (def 10), `target-class` (def 0)
- seeding: `seed-conf` (0.25), `seed-prefer-center`, `seed-delay` (frames),
  `seed-roi="x,y,w,h"` (force the initial seed, bypassing YOLO)

**nvmmfusekf**
- `target-class` (0), `det-conf` (0.25) — min YOLO confidence to fuse
- `gate-threshold` (px, def 100) — max YOLO-to-prediction center distance to fuse
- `max-lost` (def 30) — frames with no measurement before the track is dropped
- `reseed-cooldown` (def 15) — frames between upstream reseed emissions

**nvmmdrawdet**
- `draw-track` (def true) — track box + HUD; `draw-det` (def true) — raw YOLO boxes
- `draw-labels` (def true), `thickness` (px)
- `fps-smoothing` (0–1, def 0.9) — HUD FPS EMA weight
- `font-scale-divisor` (def 540) — overlay font px = `max(1, height / this)`

---

## 5. Batch eval / annotated outputs

Two helper scripts (run inside the container, `-v <workdir>:/o`):

- `run_results.sh` — runs the tracker to `fakesink` over the sample clips and
  reports `frames / fps / coverage%` (coverage = valid-track frames ÷ total; it
  is **not** accuracy vs ground truth — there is no GT). Per-frame CSV via
  `NVMMFUSEKF_CSV`.
- `encode_results.sh` — same pipeline but encodes annotated
  `results/<name>_tracked.mp4` files.

```bash
docker run --rm --runtime nvidia --network host \
  -v <repo>:/src -v <workdir>:/o -w /o gst-nvmm-infer:jp6 \
  bash -c 'bash /o/run_results.sh; bash /o/encode_results.sh'
```

---

## Notes

- Build & run **only** through Docker on the Jetson — don't install host-side.
- `nvmminfer` / `nvmmsamurai` are Jetson-only (TRT + CUDA); the pure-host
  elements (`nvmmfusekf`, etc.) build everywhere.
- Source is authored off-device and rsynced to the Jetson before each build.
