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
| 2.1 | DMA-buf export in nvmmsink | Pending |
| 2.2 | Fix nvmmappsrc polling | Pending |
| 2.3 | Dynamic shm sizing | Pending |
| 2.4 | Upstream code style | Pending |

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

## Execution order

```
1.1  Fix double-free          → test → ASAN
1.2  Fix allocator map        → test → ASAN → Jetson
1.3  Fix caps negotiation     → test → pipeline tests on Jetson
1.5  Extract ShmHeader        → test (trivial)
1.6  Unify mock/real headers  → test → Docker + Jetson
1.4  Buffer pool              → test → benchmark comparison → Jetson
2.1  DMA-buf export           → test → Jetson
2.2  Fix appsrc polling       → test → longevity test
2.3  Dynamic shm sizing       → test
2.4  Style fixes              → test
3.1  Queue/tee validation     → Jetson
3.2  Multi-consumer IPC       → TSAN + Jetson
3.3  Stress tests             → Jetson (10min longevity)
3.4  TSAN full sweep          → Docker + Jetson
3.5  ASAN full sweep          → Docker + Jetson
```

Each item: implement → unit test → integration test → sanitizer → Jetson validation.
No item is complete until all four pass.
