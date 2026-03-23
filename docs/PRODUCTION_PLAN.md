# Production Readiness Plan

Prioritized work to make gst-nvmm-cpp production-ready and upstream-viable.

## Status

| Phase | Item | Status |
|-------|------|--------|
| 1.1 | Fix double-free in nvmmconvert | Done |
| 1.2 | Fix allocator map for SURFACE_ARRAY | Done |
| 1.3 | Fix caps negotiation | Done |
| 1.4 | Implement GstNvmmBufferPool | Done |
| 1.5 | Extract ShmHeader | Done |
| 1.6 | Unify mock/real headers | Done |
| 3.1 | Queue/tee validation | Done (queue, tee x2, tee x3, queue2) |
| 3.2 | Multi-consumer IPC | Done (mid-join, producer-stop) |
| 3.3 | Stress tests | Done (state x100, 300f longevity, 500f pool) |
| 3.4 | ThreadSanitizer | Done (22 tests, no races) |
| 3.5 | AddressSanitizer | Done (22 tests, no errors) |
| 2.1 | Dynamic shm sizing | Done |
| 2.2 | Fix nvmmappsrc polling | Done |
| 2.3 | Upstream code style (GEnum, atomics, debug category) | Done |
| 4.1 | DMA-buf fd export in nvmmsink | Pending |
| 4.2 | BLOCK_LINEAR layout support | Pending |
| 4.3 | GstVideoMeta with real NVMM strides | Pending |
| 4.4 | Caps renegotiation (mid-stream resolution change) | Pending |
| 4.5 | LGPL-2.1 COPYING file | Pending |
| 4.6 | Jetson CI script | Pending |
| 4.7 | Fix allocator dimension heuristic | Pending |
| 4.8 | Format conversion pipeline test (NV12→RGBA) | Pending |

## Phase 1 — Make it actually work

### 1.1 Fix double-free in nvmmconvert

**Problem:** `gst_nvmm_convert_transform` wraps non-owned `NvBufSurface*` pointers
in `NvmmBuffer` RAII objects. When they go out of scope, the destructor calls
`NvBufSurfaceDestroy` on memory owned by the pipeline. Crash or corruption.

**Fix:** Add `NvmmBuffer::release()` that returns the raw pointer and nullifies
internal state without calling destroy. Use it in nvmmconvert after transform.

**Test:** Unit test that creates two NvmmBuffers from raw pointers, calls release(),
verifies no destroy is called. Integration test: run nvmmconvert transform on
50 consecutive frames without crash.

**Validation:** ASAN build, run full test suite.

---

### 1.2 Fix allocator map for SURFACE_ARRAY

**Problem:** `gst_nvmm_allocator_mem_map` returns plane 0 pointer but reports
`size = total_data_size`. On real NVMM, planes aren't contiguous. Writing
`total_data_size` bytes from plane 0 base = segfault.

**Fix:** Mark NVMM memory as not directly mappable via GstMemory
(`GST_MEMORY_FLAG_NOT_MAPPABLE`). Users access surfaces through
`gst_nvmm_memory_get_surface()` and the NvBufSurface API directly.
Provide a helper `gst_nvmm_memory_map_plane()` for per-plane CPU access.

**Test:** Verify `gst_memory_map()` returns FALSE on NVMM memory.
Verify `gst_nvmm_memory_map_plane(mem, plane, &data, &size)` works per-plane.
Re-enable `allocator_video_info_alloc` integration test.

**Validation:** ASAN, run on Jetson with SURFACE_ARRAY.

---

### 1.3 Fix caps negotiation in nvmmconvert

**Problem:** `transform_caps` ignores input caps and direction. No `fixate_caps`,
no `get_unit_size`. Element can't link in real pipelines.

**Fix:**
- `transform_caps`: propagate input format/resolution. If crop is set, output
  crop dimensions. If no crop, pass through input dimensions. Allow format
  changes within supported set.
- `fixate_caps`: prefer input resolution and format when multiple options exist.
- `get_unit_size`: return `GST_VIDEO_INFO_SIZE` for the given caps.
- `transform_size`: compute output size from input size and caps.

**Test:**
- Unit: verify `transform_caps` produces correct output for known inputs.
- Pipeline: `videotestsrc ! nvvidconv ! nvmmconvert crop-w=800 crop-h=600 ! nvvidconv ! jpegenc ! filesink` produces valid JPEG.
- Pipeline: `nvv4l2decoder ! nvmmconvert flip-method=2 ! nvmmsink` links and runs.
- Pipeline: format conversion NV12→RGBA through nvmmconvert.

**Validation:** All pipeline examples from README must actually work on Jetson.

---

### 1.4 Implement GstNvmmBufferPool

**Problem:** No buffer pool = every frame allocates+frees GPU memory. No
`propose_allocation` / `decide_allocation` = can't negotiate zero-copy with
upstream/downstream elements.

**Fix:**
- `GstNvmmBufferPool` subclass of `GstBufferPool`.
- `set_config`: parse `GstVideoInfo` from caps, configure NvBufSurface params.
- `alloc_buffer`: call `NvBufSurfaceCreate`, wrap in `GstBuffer` with NVMM memory.
- `free_buffer`: call `NvBufSurfaceDestroy`.
- Integrate into nvmmconvert via `decide_allocation` (allocate output pool)
  and `propose_allocation` (offer NVMM pool upstream).

**Test:**
- Allocate pool with 4 buffers, acquire/release 100 times, verify no leaks.
- Pipeline with pool: verify `NvBufSurfaceCreate` called only at startup (not per-frame).
- Stress: 1000 frames through pooled pipeline, monitor NVMM memory usage stable.

**Validation:** Compare per-frame alloc benchmark with/without pool.

---

### 1.5 Extract ShmHeader to shared header

**Problem:** `ShmHeader` copy-pasted in 5 files. Protocol changes require
editing all of them.

**Fix:** Create `gst/common/shm_protocol.h` with the struct, magic, version.
Include from sink, source, and tests.

**Test:** All existing shm-related tests pass unchanged.

---

### 1.6 Unify mock and real API headers

**Problem:** Mock struct layout (flat `mappedAddr`, nested `planeParams`) diverges
from real NVIDIA API (`mappedAddr.addr[]`, flat arrays). Bugs in real code
can't be caught by mock tests.

**Fix:** Rewrite `nvbufsurface_mock.h` to match real `nvbufsurface.h` struct layouts
exactly. Same field names, same nesting, same types. Use a single `.cpp`
implementation for both mock and real — only the header include differs.

**Test:** All mock tests pass. Verify struct sizes match between mock and real
(static_assert where possible).

**Validation:** Docker mock build + Jetson real build both pass same test suite.

---

## Phase 2 — Upstream polish

### 2.1 Implement DMA-buf fd export in nvmmsink

**Problem:** `export-dmabuf` property is stubbed. Sink always memcpy's to shm,
defeating zero-copy for IPC.

**Fix:** When `export-dmabuf=true`, write `bufferDesc` fd into ShmHeader.
Consumer uses the fd to `mmap` or import the DMA-buf directly.

**Test:** Producer exports fd, consumer opens fd, reads valid frame data.

---

### 2.2 Fix nvmmappsrc polling

**Problem:** Busy-poll with `g_usleep`, returns EOS on timeout (kills pipeline).

**Fix:** Use `GstClock` wait or condition variable. Return `GST_FLOW_FLUSHING`
when shutting down, not EOS. Make timeout configurable as a property.

**Test:** Pipeline runs for 60s with intermittent producer pauses without dying.

---

### 2.3 Dynamic shm sizing in nvmmsink

**Problem:** Always allocates 33MB regardless of resolution.

**Fix:** Size from `set_caps` — `sizeof(ShmHeader) + GST_VIDEO_INFO_SIZE(info)`.
Re-create shm on caps change.

**Test:** 480p pipeline allocates ~500KB shm, not 33MB. Multiple instances fit in 64MB Docker `/dev/shm`.

---

### 2.4 Upstream code style

- Register `GST_DEBUG_CATEGORY` per element.
- `flip-method` property as `GEnum` with named values.
- Property access mutex in nvmmconvert.
- Support `NVBUF_LAYOUT_BLOCK_LINEAR` in addition to PITCH.
- Attach `GstVideoMeta` to NVMM buffers with correct strides.

---

## Phase 3 — Advanced testing

### 3.1 GStreamer queue and tee validation

Verify elements work correctly with standard GStreamer flow control:

- **queue:** `src ! queue ! nvmmconvert ! sink` — buffering between threads.
- **queue2:** Same with `queue2` for disk-backed buffering.
- **tee:** `src ! tee name=t ! queue ! nvmmconvert ! sink1  t. ! queue ! sink2` —
  fan-out from single source to multiple consumers.
- **input-selector:** Switch between multiple sources feeding nvmmconvert.
- **output-selector:** Route nvmmconvert output to different sinks dynamically.

**Test cases:**
- tee with 2 consumers: both get correct frames, no corruption.
- tee with 3+ consumers: verify refcount handling on shared NVMM buffers.
- input-selector: switch sources mid-stream, verify no crash or leak.
- queue between decoder and nvmmconvert: verify no frame drops under load.

---

### 3.2 Multi-consumer IPC architecture

Test multiple `nvmmappsrc` consumers reading from a single `nvmmsink`:

- 1 producer, 2 consumers on same shm segment — both read correct frames.
- 1 producer, 5 consumers — verify contention handling and no data races.
- Consumer joins mid-stream — picks up from current frame, no stale data.
- Consumer leaves and rejoins — no leak, no crash on producer side.
- Producer stops while consumers are reading — graceful EOS, no segfault.

**Test cases:**
- Multi-threaded test: spawn N consumer threads, each reads M frames, verify all
  frames have correct magic/width/height/data.
- Race condition test: producer and consumer read/write ShmHeader concurrently
  for 10000 iterations, verify no torn reads.

---

### 3.3 Stress tests

- **Alloc/free stress:** 10000 NvBufSurfaceCreate/Destroy cycles, verify no
  NVMM memory leak (monitor `/proc/meminfo` or VIC allocator stats).
- **Pipeline longevity:** Run `decoder ! nvmmconvert ! nvmmsink` for 10 minutes
  continuous, verify stable RSS and no fd leak (`/proc/self/fd` count stable).
- **Rapid state changes:** NULL→READY→PLAYING→NULL 1000 times, no crash.
- **Caps renegotiation:** Change resolution mid-stream (1080p→720p→4K→480p),
  verify element handles reconfiguration.
- **OOM behavior:** Allocate NVMM buffers until allocation fails, verify
  graceful error (no crash, no leak of partially allocated resources).

---

### 3.4 Data race detection (ThreadSanitizer)

Build with `-fsanitize=thread` and run:

- All unit tests under TSAN.
- Multi-threaded IPC test (producer + consumer threads).
- Pipeline with `queue` elements (introduces thread boundaries).
- Property changes from main thread while pipeline is PLAYING.
- `nvmmsink` render from streaming thread + property reads from app thread.

**Known race candidates:**
- `ShmHeader.ready` / `frame_number` — currently uses `__sync_synchronize` but
  should use `std::atomic` with proper memory ordering.
- nvmmconvert crop/flip properties — no locking between set_property and transform.
- nvmmappsrc `last_frame_number` — read in create(), could race with state changes.

---

### 3.5 AddressSanitizer validation

Build with `-fsanitize=address` and run:

- Full test suite — catch any heap/stack overflow, use-after-free.
- Integration tests with real NVMM (map/unmap cycles).
- Buffer pool acquire/release stress test.
- Pipeline shutdown during active streaming (verify no dangling pointers).

---

## Phase 4 — Production hardening

### 4.1 DMA-buf fd export in nvmmsink

**Problem:** `export-dmabuf` property is stubbed — always writes -1. The "zero-copy
IPC" claim requires fd sharing so consumers can import the DMA-buf without memcpy.

**Fix:** When `export-dmabuf=true` and buffer is NVMM, write `bufferDesc` (the
DMA-buf fd) into ShmHeader. Consumer uses the fd to mmap or import directly.
For non-NVMM buffers, fall back to memcpy as currently.

**Test:** Producer exports fd, standalone C consumer opens fd, verifies non-negative
and readable. Pipeline test: producer with `export-dmabuf=true`, consumer verifies
`header->dmabuf_fd >= 0`.

---

### 4.2 BLOCK_LINEAR layout support

**Problem:** `NvBufSurfaceCreate` hardcodes `NVBUF_LAYOUT_PITCH`. NVIDIA decoders
(`nvv4l2decoder`) output `NVBUF_LAYOUT_BLOCK_LINEAR` surfaces. Passing block-linear
surfaces into `NvBufSurfTransform` where the output is pitch-linear should work
(VIC handles layout conversion), but we never tested it.

**Fix:** Accept block-linear input surfaces in nvmmconvert. When we allocate output
via the pool, use PITCH layout (VIC can convert). Test with real decoder output.

**Test:** `filesrc ! h264parse ! nvv4l2decoder ! nvmmconvert flip-method=2 ! nvmmsink`
produces correct output with no garbling. Compare with `nvvidconv` baseline.

---

### 4.3 GstVideoMeta with real NVMM strides

**Problem:** Buffer pool attaches `GstVideoMeta` using `GstVideoInfo` strides, but
NVMM surfaces may have different actual strides (alignment, padding). Downstream
elements that use the meta get wrong plane offsets.

**Fix:** After `NvBufSurfaceCreate`, read actual strides from `planeParams.pitch[]`
and `planeParams.offset[]`, use those in `gst_buffer_add_video_meta_full()`.

**Test:** Verify `GstVideoMeta` strides match `NvBufSurface` planeParams on Jetson
for NV12 1080p and RGBA 720p.

---

### 4.4 Caps renegotiation (mid-stream resolution change)

**Problem:** If upstream changes resolution, `set_caps` recreates the buffer pool
but doesn't drain outstanding buffers. Can crash or leak.

**Fix:** In `set_caps`, deactivate old pool (which blocks until all buffers return),
then create new pool. Handle the case where `set_caps` is called from the streaming
thread while buffers are in-flight.

**Test:** Pipeline that changes resolution every 30 frames
(1080p → 720p → 480p → 1080p), verify no crash or leak over 200 frames.

---

### 4.5 LGPL-2.1 COPYING file

Add `COPYING` with LGPL-2.1-or-later text. Update `meson.build` license field.

---

### 4.6 Jetson CI script

Create `scripts/jetson-test.sh` that:
1. Builds on Jetson
2. Runs all unit tests
3. Runs pipeline tests (passthrough, flip, scale, crop, combined)
4. Runs benchmarks
5. Reports results in CI-parseable format

Document SSH-based invocation from GitHub Actions or local.

---

### 4.7 Fix allocator dimension heuristic

**Problem:** `gst_allocator_alloc(alloc, size, NULL)` guesses dimensions from byte
count assuming NV12. Wrong for RGBA. Wrong for non-standard resolutions.

**Fix:** Add `gst_nvmm_allocator_alloc_video(alloc, video_info)` that takes explicit
`GstVideoInfo`. The byte-size heuristic stays as fallback but logs a warning.

---

### 4.8 Format conversion pipeline test (NV12→RGBA)

Test `nvmmconvert` doing color space conversion in a real pipeline:
```
videotestsrc ! nvvidconv ! NVMM NV12 ! nvmmconvert ! NVMM RGBA ! nvvidconv ! jpegenc ! file
```
Verify output is valid RGBA JPEG with correct colors.

---

## Execution order

```
Phase 1 (done): 1.1-1.6
Phase 2 (done): 2.1-2.3
Phase 3 (done): 3.1-3.5
Phase 4:
  4.5  COPYING file             → trivial
  4.8  Format conversion test   → pipeline test on Jetson
  4.2  BLOCK_LINEAR layout      → test with real decoder on Jetson
  4.3  GstVideoMeta strides     → test on Jetson
  4.1  DMA-buf fd export        → test → Jetson
  4.7  Allocator video_info API → test → Docker + Jetson
  4.4  Caps renegotiation       → test → stress on Jetson
  4.6  Jetson CI script         → test
```

Each item: implement → unit test → integration test → Jetson validation.
No item is complete until all pass.
