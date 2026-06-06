# Jetson hardware validation

Validated on two Jetson platforms (both in Docker and native):

- **Jetson Xavier NX** — JetPack 5.1 (L4T R35.x), GStreamer 1.16.3
- **Jetson Orin NX** — JetPack 6 (L4T R36.x), GStreamer 1.20.3

## Test results

All 7 test suites pass on both Xavier NX and Orin NX:

```
 1/7 nvmm_buffer        OK   10 passed   (create, map, move, release, export_fd, planes)
 2/7 nvmm_transform     OK    9 passed   (scale, crop, convert, flip, rotate 90/270, interpolation, null safety)
 3/7 gst_nvmm_allocator OK    8 passed   (create, alloc, surface map, per-plane, roundtrip)
 4/7 nvmm_sink          OK    4 passed   (create, properties, state, shm lifecycle)
 5/7 nvmm_appsrc        OK    2 passed   (create, properties)
 6/7 gstcheck_elements  OK    8 passed   (discovery, state, properties, caps, pipeline)
 7/7 integration        OK    6 passed   (multi-shm, dynamic props, pipeline bin, alloc stress, protocol, missing-shm)
Ok: 7   Fail: 0
```

11 pipeline tests also pass via `scripts/jetson-test.sh`:
passthrough, flip-180, rotate-90, rotate-270, scale, crop, format-convert,
decoder, tee-2way, 30f-throughput, and the two-process IPC pipeline
(`nvmmsink` → `nvmmappsrc`, verified frames cross the process boundary).

## Stress tests

| Test | Result |
|------|--------|
| State changes x100 (NULL→READY→NULL) | PASS |
| 500f pool stress (1080p→720p, flip) | PASS (21s) |
| 50 rapid pool recreate cycles | PASS |
| tee x3 with different transforms | PASS |
| Caps renegotiation (4 resolution changes) | PASS |

## Sanitizer results

| Sanitizer | Tests | Result |
|-----------|-------|--------|
| AddressSanitizer | 22 (buffer + transform + allocator) | Clean |
| ThreadSanitizer | 22 (buffer + transform + allocator) | Clean |

## Benchmark results

1000 iterations each. VIC transform includes hardware sync.

**Xavier NX (JetPack 5.1)**

| Operation | Resolution | Avg (us) | Min (us) | Max (us) |
|-----------|-----------|----------|----------|----------|
| alloc/free | NV12 1080p | 591 | 128 | 2291 |
| alloc/free | RGBA 1080p | 2095 | 1072 | 2129 |
| alloc/free | NV12 4K | 3245 | 2091 | 2701 |
| alloc/free | RGBA 4K | 10104 | 7110 | 9164 |
| map/unmap | NV12 1080p | 231 | 222 | 493 |
| VIC transform | 1080p -> 480p | **1947** | 1577 | 4752 |
| VIC transform | 1080p -> 720p | **1655** | 1594 | 1826 |
| VIC transform | 4K -> 1080p | **4002** | 3938 | 4913 |

**Orin NX (JetPack 6)**

| Operation | Resolution | Avg (us) | Min (us) | Max (us) |
|-----------|-----------|----------|----------|----------|
| alloc/free | NV12 1080p | 117 | 14 | 1551 |
| alloc/free | RGBA 1080p | 366 | 33 | 1072 |
| map/unmap | NV12 1080p | 298 | 275 | 374 |
| map/unmap | NV12 480p | 49 | 39 | 61 |
| VIC transform | 1080p -> 480p | **35** | 27 | 49 |
| VIC transform | 1080p -> 720p | **95** | 85 | 114 |
| VIC transform | 4K -> 1080p | **285** | 217 | 459 |
| VIC transform | 4K -> 480p | **31** | 26 | 67 |

Orin allocation is **5x faster** than Xavier NX. VIC transform **14-56x faster**
depending on resolution (e.g. 1080p->480p: 1947 us -> 35 us).

## VIC hardware accelerator verification

Evidence that the Tegra VIC (Video Image Compositor) hardware engine is engaged:

1. **NvBufSurfTransform defaults to VIC compute on Jetson** — the API selects `NvBufSurfTransformCompute_Default`, which maps to VIC on Tegra (not GPU or CPU).
2. **Transform latency confirms hardware acceleration** — 35 us per 1080p-to-480p scale on Orin NX (table above). A CPU scale at 1080p would take milliseconds; ~28,500 FPS is only achievable via dedicated hardware.
3. **NVMM SURFACE_ARRAY memory type** — tests use `NVBUF_MEM_DEFAULT` → `NVBUF_MEM_SURFACE_ARRAY` on Jetson (physically contiguous, VIC/NVDEC-managed). Tests FAIL with `NVBUF_MEM_SYSTEM` for hardware-only operations, proving the hardware path.
4. **DMA-buf fd export works** — `export_fd()` returns a valid fd from `bufferDesc`.
5. **VIC device node** — `/dev/nvhost-vic` is present and accessible.

## Transfer path verification

| Path | Pipeline | Result |
|------|----------|--------|
| **CPU -> GPU** | `videotestsrc ! nvvidconv ! NVMM ! nvmmsink` | OK |
| **GPU -> GPU** | `nvv4l2decoder(NVMM) ! nvvidconv ! NVMM(scaled) ! nvmmsink` | OK |
| **GPU -> CPU** | `nvv4l2decoder(NVMM) ! nvvidconv ! x-raw ! jpegenc ! file` | OK |

## Resolution verification

| Resolution | Alloc | Map | Transform (to 480p) | Pipeline |
|------------|-------|-----|---------------------|----------|
| **FHD** 1920x1080 | 3103 us | 77 us | 5324 us | OK (133 KB JPEG) |
| **4K** 3840x2160 | 263 us | 105 us | 17028 us | OK (491 KB JPEG) |

NvmmBuffer API at both resolutions: NV12 = 2 planes (Y+UV); FHD data_size
3,407,872 B; 4K data_size 12,582,912 B; DMA-buf fd export works at both.

## nvmmconvert pipeline proof

All operations verified via `gst-launch-1.0` on Jetson Xavier NX:

| Operation | Output |
|-----------|--------|
| Passthrough | ![passthrough](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/convert_passthrough.jpg) |
| Flip 180° | ![flip180](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/convert_flip180.jpg) |
| Rotate 90° CW (640×480→480×640) | ![rotate90](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/convert_rotate90.jpg) |
| Rotate 270° CCW (640×480→480×640) | ![rotate270](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/convert_rotate270.jpg) |
| Flip horizontal | ![flipH](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/convert_flipH.jpg) |
| Scale 1080p→480p | ![scale](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/convert_scale.jpg) |
| Crop (100,50,800,600) | ![crop](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/convert_crop.jpg) |

## Test outputs

All images generated on Jetson Xavier NX with real NVMM hardware:

| Image | Description |
|-------|-------------|
| ![smpte](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/smpte_1080p.jpg) | **smpte_1080p.jpg** — 1920x1080 SMPTE test pattern |
| ![gpu2cpu](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/gpu2cpu_1080p.jpg) | **gpu2cpu_1080p.jpg** — 1080p GPU→CPU transfer |
| ![4k](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/4k_roundtrip.jpg) | **4k_roundtrip.jpg** — 3840x2160 CPU→NVMM→CPU |
| ![ipc480](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/ipc_480p.jpg) | **ipc_480p.jpg** — IPC consumer via nvmmsink→shm→nvmmappsrc |
| ![shm](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/shm_consumer_frame.jpg) | **shm_consumer_frame.jpg** — standalone C shm reader (ROS2-style) |

## Reproducing on Jetson

```bash
git clone https://github.com/PavelGuzenfeld/gst-nvmm-cpp.git
cd gst-nvmm-cpp

# Build (CMake; meson equivalent in Getting started)
cmake -S . -B build-cmake
cmake --build build-cmake -j$(nproc)

# Test + benchmark
ctest --test-dir build-cmake --output-on-failure
./build-cmake/benchmarks/bench_nvmm

# Install + use
sudo cmake --install build-cmake
rm -f ~/.cache/gstreamer-1.0/registry.*.bin   # force a GStreamer rescan
gst-inspect-1.0 nvmmconvert
```
