# Creating a new element

The suite's elements follow a small number of repeating patterns. This page
walks through them using `gst/nvmmexample/gstnvmmdetlog.cpp` — a complete,
buildable element that ships in the tree (compiled by CI on every platform,
never installed). Copy it as the starting point for a new element.

## Pick an archetype

| Archetype | Base class | Pixels touched | Builds on | Template in tree |
|---|---|---|---|---|
| Metadata passthrough | `GstBaseTransform` (in place) | no | everywhere | `gst/nvmmexample`, `gst/nvmmtracker` |
| VIC processing | `GstBaseTransform` | VIC only | Jetson | `gst/nvmmconvert` |
| Multi-input aggregation | `GstAggregator` | depends | everywhere / Jetson | `gst/nvmmfusion`, `gst/nvmmcompositor` |
| Engine inference | `GstBaseTransform` (in place) | CUDA/VIC/engine | Jetson (gated) | `gst/nvmminfer`, `gst/nvmmsecondaryinfer` |

Most analytics nodes are metadata passthroughs: the NVMM frame flows through
untouched and the element only reads or attaches `GstMeta`. Because the pixels
are never mapped, such elements compile and unit-test on the x86 mock build.

## Anatomy of `nvmmdetlog`

The example is an in-place passthrough that logs the detection and
classification metadata. Its parts, in source order:

1. **Type declaration** — `G_DECLARE_FINAL_TYPE` + a plain struct deriving
   from `GstBaseTransform`. C++ members that need constructors are stored as
   pointers and created in `_init`/freed in `finalize` (GObject allocates the
   struct with `malloc`; see `gstnvmminfer.cpp` for the pattern with
   `std::vector`/`std::string` members).
2. **Pad templates** — identical `video/x-raw(memory:NVMM), format=NV12` caps
   on both pads. An element that changes resolution or format instead
   implements `transform_caps` (see `gstnvmmconvert.cpp`) — and then must also
   implement `transform_size` so output buffers are sized from the *output*
   caps (`gstnvmmdrawdet.cpp` shows why: an NVMM gst-buffer's size is a
   surface handle, not a pixel count).
3. **`transform_ip`** — the per-buffer hook. Metadata is read with
   `gst_buffer_get_nvmm_det_meta(buf)`; absence of metadata is not an error
   for an analytics node — log and return `GST_FLOW_OK`.
4. **Properties** — plain GObject `set_property`/`get_property` plus
   `g_object_class_install_property` in `class_init`.
5. **Class wiring** — pad templates, element metadata, vmethods, and a
   `GST_DEBUG_CATEGORY` named after the element.
6. **Plugin entry point** — `GST_PLUGIN_DEFINE` with `gst_element_register`.
   One plugin per element directory keeps `GST_PLUGIN_PATH` composition simple.

In `_init`, an in-place passthrough calls
`gst_base_transform_set_in_place(..., TRUE)` so GStreamer hands `transform_ip`
the buffer without copying pixels.

## Reaching the pixels

When the element does touch the frame, get the `NvBufSurface` from the buffer:

```c
static NvBufSurface *surface_of(GstBuffer *buf) {
    GstMemory *m = gst_buffer_peek_memory(buf, 0);
    if (m && gst_is_nvmm_memory(m))
        return gst_nvmm_memory_get_surface(m);
    /* fallback: NVMM buffers from stock elements map to the surface struct */
    GstMapInfo map;
    if (m && gst_buffer_map(buf, &map, GST_MAP_READ)) {
        NvBufSurface *s = (NvBufSurface *)map.data;
        gst_buffer_unmap(buf, &map);
        return s;
    }
    return NULL;
}
```

Process it with `NvBufSurfTransform` (VIC — crop/scale/convert, see
`nvmm_transform.cpp`) or map it into CUDA via EGL interop
(`nvmminfer/preprocess.cpp`). NPP calls must use the `_Ctx` API bound to the
element's own CUDA stream — the legacy `nppSetStream` is process-global and
collides with other elements in the same pipeline.

## Attaching your own metadata

New result types are added as *sibling metas*: a separate `GstMeta` whose
entries align by index with the det meta's objects array on the same buffer.
`gst/common/nvmm_class_meta.{h,cpp}` is the template — registration via
`nvmm_meta_api_register_once`, a deep-copying copy-transform, and an
`add` helper. Two rules:

- Implement the meta in `gst/common` (it compiles into the shared
  `nvmm_common` library; a static copy per plugin would race the GType
  registration across threads).
- The copy-transform must return `FALSE` for non-copy transforms, so the
  index pairing with the det meta can never go stale.

## Splitting out a host-testable core

Algorithm logic lives in a dependency-free class next to the element, not in
the GObject code: `nvmmtracker/tracker.cpp`, `nvmmsecondaryinfer/
secondary_cache.cpp`, `common/nvmm_motion.cpp`. The matching unit test in
`tests/` compiles that file directly and runs on the x86 CI build:

```meson
test_my_core = executable('test_my_core',
  'test_my_core.cpp',
  '../gst/mynewelement/my_core.cpp',
  include_directories : [config_inc, common_inc,
                         include_directories('../gst/mynewelement')],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)
test('my_core', test_my_core, protocol : 'exitcode')
```

Tests use the self-registering harness in `tests/test_harness.h`
(`TEST(...)` + `ASSERT_*`; provide a `main` that prints the counters).

## Build wiring

`gst/mynewelement/meson.build`:

```meson
gstmynewelement = library('gstmynewelement',
  files('gstmynewelement.cpp', 'my_core.cpp'),
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep, gst_base_dep],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)
```

Then add `subdir('gst/mynewelement')` to the top-level `meson.build`:

- unconditionally for pure-host elements (next to `nvmmtracker`);
- inside `if have_tensorrt and have_nvbufsurface` for engine elements, or
  `if have_vpi and have_nvbufsurface` for VPI — hardware-only elements are
  *skipped*, never stubbed, on the mock build.

## Checklist

- [ ] element compiles with `-Dwerror=true` on the mock build
      (`docker build -f docker/Dockerfile.dev` runs exactly this)
- [ ] `gst-inspect-1.0 mynewelement` shows the element and its properties
- [ ] algorithm core has a host unit test registered in `tests/meson.build`
- [ ] per-frame failures log a warning and return `GST_FLOW_OK`; only
      misconfiguration fails `start()` with `GST_ELEMENT_ERROR`
- [ ] a docs page under `docs/elements/` and an entry in `mkdocs.yml`

Verifying the example end to end:

```bash
docker build -f docker/Dockerfile.dev -t gst-nvmm-cpp:dev .
docker run --rm gst-nvmm-cpp:dev bash -c \
  'GST_PLUGIN_PATH=/src/builddir/gst/nvmmexample \
   LD_LIBRARY_PATH=/src/builddir/gst/common \
   gst-inspect-1.0 nvmmdetlog'
```

On a Jetson, drop it into any pipeline that carries detection metadata:

```bash
gst-launch-1.0 ... ! nvmminfer engine-file=yolo.engine ! nvmmtracker \
  ! nvmmdetlog per-object=true ! fakesink
```
