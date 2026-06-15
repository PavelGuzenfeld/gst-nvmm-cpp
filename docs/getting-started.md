# Getting started

## Requirements

- NVIDIA Jetson (Xavier or Orin), JetPack 5.1.1+ (JP5) or JP6, GStreamer ≥ 1.16.
- For host-side unit testing: any x86_64 with Docker (uses a mock NvBufSurface API).

## Build (Jetson, native)

```bash
meson setup builddir -Dcpp_std=c++14 -Dbuildtype=debugoptimized -Dwerror=false
ninja -C builddir
```

The build auto-detects the real NVMM stack via `pkg-config`/library search. If
`NvBufSurface` isn't found it errors unless you opt into the mock with
`-Dmock=true` (host testing only — never production).

### Build-time options (CUDA targets)

The CUDA elements (`nvmmsamurai` and its probes) take two combo options so a bad
value is rejected at `meson setup` rather than deep inside `nvcc`:

| Option | Values | Default | Meaning |
|--------|--------|---------|---------|
| `gpu_arch` | `sm_72`…`sm_90` | `sm_87` | GPU arch — `sm_87` = Orin, `sm_72` = Xavier; a mismatch makes kernels fail to launch |
| `gpu_cxx_std` | `c++17`, `c++20` | `c++20` | C++ standard for the CUDA kernels (`nvcc -std`); drop to `c++17` only for an older toolchain |

```bash
meson setup builddir -Dgpu_arch=sm_87 -Dgpu_cxx_std=c++20
```

## Build & test (host, Docker, mock API)

```bash
docker build -f docker/Dockerfile.dev -t gst-nvmm-cpp:dev .
docker run --rm --shm-size=256m gst-nvmm-cpp:dev meson test -C builddir --verbose
```

## Run the elements

Point GStreamer at the built plugins:

```bash
export GST_PLUGIN_PATH=$PWD/builddir/gst/nvmmconvert:$PWD/builddir/gst/nvmmsink:\
$PWD/builddir/gst/nvmmappsrc:$PWD/builddir/gst/nvmmalloc
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:$GST_PLUGIN_PATH

gst-inspect-1.0 nvmmconvert
```

A minimal VIC transform pipeline (CPU → NVMM → scale → back to CPU → JPEG):

```bash
gst-launch-1.0 videotestsrc num-buffers=1 pattern=smpte ! \
  'video/x-raw,width=1920,height=1080,format=I420' ! \
  nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
  nvmmconvert ! 'video/x-raw(memory:NVMM),width=640,height=480' ! \
  nvvidconv ! 'video/x-raw,format=I420' ! nvjpegenc ! filesink location=out.jpg
```

> `nvmmconvert`/`nvmmappsrc` consume and emit `video/x-raw(memory:NVMM)`. Use
> `nvvidconv` (VIC) — **not** `videoconvert` (CPU) — to cross the NVMM boundary.

## Next steps

- [Pipeline examples](pipelines.md) — transport, processing, encoding,
  multi-camera fan-out.
- [Inference graphs](inference-graphs.md) — detector / tracker / classifier /
  optical-flow compositions.
- [Creating a new element](extending.md) — extend the suite with your own node.
