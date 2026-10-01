#include "config.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include "gstnvmmallocator.h"
#include "gstnvmmbufferpool.h"
#include "nvmm_class_meta.h"
#include "nvmm_det_meta.h"
#include "nvmm_optical_flow_meta.h"
#include "nvmm_track_meta.h"
#include "shm_protocol.h"

#ifdef NVMM_MOCK_API
#include "nvbufsurface_mock.h"
#else
#include <nvbufsurface.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "test_harness.h"

namespace {

/// libnvbufsurftransform's fini deletes pthread key 0, which it never created. Under TSan key 0 is
/// the thread-finalize key, so the fini's join of its own worker spins forever (#99).
#if defined(__SANITIZE_THREAD__) && HAVE_NVBUFSURFTRANSFORM
constexpr bool exit_hangs_in_nvbufsurftransform_fini = true;
#else
constexpr bool exit_hangs_in_nvbufsurftransform_fini = false;
#endif

/// Declared first: within one TU, static objects initialize in declaration order.
struct GstInit { GstInit() { gst_init(nullptr, nullptr); } } _gst_init;

struct NvmmVideoMemory {
    GstAllocator *alloc = gst_nvmm_allocator_new(0);
    GstMemory *mem;
    NvmmVideoMemory(GstVideoFormat fmt, guint w, guint h)
        : mem(gst_nvmm_allocator_alloc_video(alloc, fmt, w, h)) {}
    ~NvmmVideoMemory() {
        if (mem) gst_memory_unref(mem);
        gst_object_unref(alloc);
    }
};

TEST(alloc_video_nv12_and_rgba) {
    NvmmVideoMemory nv12(GST_VIDEO_FORMAT_NV12, 1920, 1080);
    ASSERT_TRUE(GST_IS_NVMM_ALLOCATOR(nv12.alloc));
    ASSERT_NOT_NULL(nv12.mem);
    ASSERT_TRUE(gst_is_nvmm_memory(nv12.mem));
    ASSERT_NOT_NULL(gst_nvmm_memory_get_surface(nv12.mem));

    NvmmVideoMemory rgba(GST_VIDEO_FORMAT_RGBA, 1280, 720);
    ASSERT_NOT_NULL(rgba.mem);
    ASSERT_TRUE(gst_is_nvmm_memory(rgba.mem));
    ASSERT_TRUE(rgba.mem->size > 0);
}

TEST(alloc_video_zero_size_fails) {
    NvmmVideoMemory m(GST_VIDEO_FORMAT_NV12, 0, 0);
    ASSERT_TRUE(m.mem == NULL);
}

TEST(direct_map_returns_surface) {
    NvmmVideoMemory m(GST_VIDEO_FORMAT_NV12, 640, 480);
    ASSERT_NOT_NULL(m.mem);

    GstMapInfo map_info;
    ASSERT_TRUE(gst_memory_map(m.mem, &map_info, GST_MAP_READ));
    ASSERT_NOT_NULL(map_info.data);
    ASSERT_TRUE(map_info.data == gst_nvmm_memory_get_surface(m.mem));
    gst_memory_unmap(m.mem, &map_info);
}

TEST(per_plane_write_read_roundtrip) {
    NvmmVideoMemory m(GST_VIDEO_FORMAT_RGBA, 64, 64);
    ASSERT_NOT_NULL(m.mem);

    guint8 *data = NULL;
    gsize size = 0;
    ASSERT_TRUE(gst_nvmm_memory_map_plane(m.mem, 0, GST_MAP_WRITE, &data, &size));
    ASSERT_NOT_NULL(data);
    ASSERT_TRUE(size > 0);
    memset(data, 0xCD, size);
    gst_nvmm_memory_unmap_plane(m.mem);

    ASSERT_TRUE(gst_nvmm_memory_map_plane(m.mem, 0, GST_MAP_READ, &data, &size));
    ASSERT_TRUE(data[0] == 0xCD);
    gst_nvmm_memory_unmap_plane(m.mem);
}

TEST(non_nvmm_memory_rejected) {
    GstAllocator *sys_alloc = gst_allocator_find(GST_ALLOCATOR_SYSMEM);
    GstMemory *mem = gst_allocator_alloc(sys_alloc, 1024, NULL);
    ASSERT_TRUE(!gst_is_nvmm_memory(mem));
    ASSERT_TRUE(gst_nvmm_memory_get_surface(mem) == NULL);

    guint8 *data = NULL;
    gsize size = 0;
    ASSERT_TRUE(!gst_nvmm_memory_map_plane(mem, 0, GST_MAP_READ, &data, &size));

    gst_memory_unref(mem);
}

/// Jetson pitch alignment makes the real strides differ from GstVideoInfo's
/// defaults (640-wide NV12 has a 768 pitch).
TEST(pool_video_meta_real_strides) {
    GstBufferPool *pool = gst_nvmm_buffer_pool_new();
    ASSERT_NOT_NULL(pool);

    GstCaps *caps = gst_caps_from_string(
        "video/x-raw(memory:NVMM), format=(string)NV12, "
        "width=(int)1920, height=(int)1080");
    GstVideoInfo vinfo;
    ASSERT_TRUE(gst_video_info_from_caps(&vinfo, caps));

    GstStructure *config = gst_buffer_pool_get_config(pool);
    gst_buffer_pool_config_set_params(config, caps,
        (guint)GST_VIDEO_INFO_SIZE(&vinfo), 2, 4);
    ASSERT_TRUE(gst_buffer_pool_set_config(pool, config));
    ASSERT_TRUE(gst_buffer_pool_set_active(pool, TRUE));

    GstBuffer *buf = NULL;
    ASSERT_TRUE(gst_buffer_pool_acquire_buffer(pool, &buf, NULL) == GST_FLOW_OK);
    ASSERT_NOT_NULL(buf);

    GstVideoMeta *vmeta = gst_buffer_get_video_meta(buf);
    ASSERT_NOT_NULL(vmeta);

    void *surface = gst_nvmm_memory_get_surface(gst_buffer_peek_memory(buf, 0));
    ASSERT_NOT_NULL(surface);
    NvBufSurface *nvsurf = static_cast<NvBufSurface *>(surface);
    NvBufSurfacePlaneParams &pp = nvsurf->surfaceList[0].planeParams;

    for (guint i = 0; i < vmeta->n_planes; i++) {
        ASSERT_TRUE(vmeta->stride[i] == (gint)pp.pitch[i]);
        ASSERT_TRUE(vmeta->offset[i] == (gsize)pp.offset[i]);
    }

    gst_buffer_unref(buf);
    gst_buffer_pool_set_active(pool, FALSE);
    gst_object_unref(pool);
    gst_caps_unref(caps);
}

/// NO_SHARE would make make_writable deep-copy, so tee fan-out would stop being zero-copy.
TEST(tee_make_writable_zero_copy) {
    NvmmVideoMemory m(GST_VIDEO_FORMAT_NV12, 640, 480);
    ASSERT_NOT_NULL(m.mem);
    ASSERT_TRUE(!GST_MEMORY_FLAG_IS_SET(m.mem, GST_MEMORY_FLAG_NO_SHARE));
    void *surf0 = gst_nvmm_memory_get_surface(m.mem);

    GstBuffer *buf = gst_buffer_new();
    gst_buffer_append_memory(buf, m.mem);
    m.mem = nullptr;

    GstBuffer *buf2 = gst_buffer_copy(buf);
    ASSERT_TRUE(gst_nvmm_memory_get_surface(gst_buffer_peek_memory(buf2, 0)) == surf0);

    gst_buffer_unref(buf2);
    gst_buffer_unref(buf);
}

TEST(share_references_same_surface) {
    NvmmVideoMemory m(GST_VIDEO_FORMAT_NV12, 640, 480);
    ASSERT_NOT_NULL(m.mem);

    GstMemory *shared = gst_memory_share(m.mem, 0, -1);
    ASSERT_NOT_NULL(shared);
    ASSERT_TRUE(gst_is_nvmm_memory(shared));
    ASSERT_TRUE(gst_nvmm_memory_get_surface(shared) == gst_nvmm_memory_get_surface(m.mem));

    gst_memory_unref(shared);
}

TEST(share_outlives_parent) {
    NvmmVideoMemory m(GST_VIDEO_FORMAT_NV12, 640, 480);
    ASSERT_NOT_NULL(m.mem);
    void *surf = gst_nvmm_memory_get_surface(m.mem);

    GstMemory *shared = gst_memory_share(m.mem, 0, -1);
    ASSERT_NOT_NULL(shared);
    gst_memory_unref(m.mem);
    m.mem = nullptr;

    ASSERT_TRUE(gst_nvmm_memory_get_surface(shared) == surf);
    gst_memory_unref(shared);
}

/// Each cell is (dx, dy) as int16 S10.5: px = raw / 32.
GstMemory *make_flow(gint w, gint h, int16_t dx, int16_t dy) {
    GstMemory *mem = gst_allocator_alloc(nullptr, (gsize)w * h * 4, nullptr);
    GstMapInfo map;
    gst_memory_map(mem, &map, GST_MAP_WRITE);
    int16_t *v = reinterpret_cast<int16_t *>(map.data);
    for (gint i = 0; i < w * h; i++) { v[2 * i] = dx; v[2 * i + 1] = dy; }
    gst_memory_unmap(mem, &map);
    return mem;
}

TEST(optical_flow_add_and_get) {
    ASSERT_TRUE(nvmm_optical_flow_meta_get_info() != nullptr);
    GstBuffer *buf = gst_buffer_new();
    GstMemory *flow = make_flow(16, 12, 64, -48);
    NvmmOpticalFlowMeta *m = gst_buffer_add_nvmm_optical_flow_meta(
        buf, flow, 16, 12, 4, 64, 48);
    gst_memory_unref(flow);
    ASSERT_TRUE(m != nullptr);

    NvmmOpticalFlowMeta *got = gst_buffer_get_nvmm_optical_flow_meta(buf);
    ASSERT_TRUE(got == m);
    ASSERT_TRUE(got->mv_width == 16 && got->mv_height == 12);
    ASSERT_TRUE(got->grid_size == 4);
    ASSERT_TRUE(got->frame_width == 64 && got->frame_height == 48);
    GstMapInfo map;
    ASSERT_TRUE(gst_memory_map(got->mv, &map, GST_MAP_READ));
    const int16_t *v = reinterpret_cast<const int16_t *>(map.data);
    const double dx = v[0] / 32.0, dy = v[1] / 32.0;
    gst_memory_unmap(got->mv, &map);
    ASSERT_TRUE(dx == 2.0 && dy == -1.5);
    gst_buffer_unref(buf);
}

TEST(optical_flow_copy_transform) {
    GstBuffer *buf = gst_buffer_new();
    GstMemory *flow = make_flow(8, 8, 32, 0);
    gst_buffer_add_nvmm_optical_flow_meta(buf, flow, 8, 8, 4, 32, 32);
    gst_memory_unref(flow);

    GstBuffer *copy = gst_buffer_copy(buf);
    NvmmOpticalFlowMeta *cm = gst_buffer_get_nvmm_optical_flow_meta(copy);
    ASSERT_TRUE(cm != nullptr);
    ASSERT_TRUE(cm->mv_width == 8 && cm->mv_height == 8 && cm->grid_size == 4);
    ASSERT_TRUE(cm->mv != nullptr);

    gst_buffer_unref(buf);
    GstMapInfo map;
    ASSERT_TRUE(gst_memory_map(cm->mv, &map, GST_MAP_READ));
    ASSERT_TRUE(reinterpret_cast<const int16_t *>(map.data)[0] == 32);
    gst_memory_unmap(cm->mv, &map);
    gst_buffer_unref(copy);
}

void fill_frame(NvmmFrameMeta *f, guint32 n) {
    memset(f, 0, sizeof(*f));
    f->frame_number = 42;
    f->infer_width = 1920;
    f->infer_height = 1080;
    f->num_objects = n;
    for (guint32 i = 0; i < n && i < NVMM_META_MAX_OBJECTS; i++) {
        f->objects[i].left = (float)i;
        f->objects[i].top = (float)(i * 2);
        f->objects[i].width = 10.0f + i;
        f->objects[i].height = 20.0f + i;
        f->objects[i].class_id = (int)i;
        f->objects[i].confidence = 0.5f + (float)i / 1000.0f;
        f->objects[i].tracker_id = 1000 + i;
        snprintf(f->objects[i].label, NVMM_META_LABEL_LEN, "class_%u", i);
    }
}

TEST(shm_segment_layout) {
    ASSERT_EQ(nvmm_shm_segment_size(0), sizeof(NvmmShmHeader));
    ASSERT_EQ(nvmm_shm_segment_size(1),
              sizeof(NvmmShmHeader) + (size_t)NVMM_POOL_SIZE * sizeof(NvmmFrameMeta));

    unsigned char *base = (unsigned char *)g_malloc0(nvmm_shm_segment_size(1));
    NvmmFrameMeta *slot0 = nvmm_shm_meta(base, 0);
    NvmmFrameMeta *slot3 = nvmm_shm_meta(base, 3);
    ASSERT_EQ((unsigned char *)slot0, base + sizeof(NvmmShmHeader));
    ASSERT_EQ((size_t)((unsigned char *)slot3 - (unsigned char *)slot0),
              3u * sizeof(NvmmFrameMeta));
    g_free(base);
}

TEST(det_meta_add_get_roundtrip) {
    NvmmFrameMeta f;
    fill_frame(&f, 3);

    GstBuffer *buf = gst_buffer_new();
    ASSERT_TRUE(gst_buffer_add_nvmm_det_meta(buf, &f) != nullptr);

    GstNvmmDetMeta *got = gst_buffer_get_nvmm_det_meta(buf);
    ASSERT_TRUE(got != nullptr);
    ASSERT_EQ(got->frame_number, 42u);
    ASSERT_EQ(got->infer_width, 1920u);
    ASSERT_EQ(got->infer_height, 1080u);
    ASSERT_EQ(got->num_objects, 3u);
    ASSERT_TRUE(got->objects != nullptr);
    ASSERT_EQ(got->objects[2].class_id, 2);
    ASSERT_EQ(got->objects[2].tracker_id, 1002u);
    ASSERT_TRUE(got->objects[1].width == 11.0f);
    ASSERT_TRUE(strcmp(got->objects[2].label, "class_2") == 0);

    gst_buffer_unref(buf);
}

TEST(det_meta_empty_detections) {
    NvmmFrameMeta f;
    fill_frame(&f, 0);
    GstBuffer *buf = gst_buffer_new();
    GstNvmmDetMeta *m = gst_buffer_add_nvmm_det_meta(buf, &f);
    ASSERT_TRUE(m != nullptr);
    ASSERT_EQ(m->num_objects, 0u);
    ASSERT_TRUE(m->objects == nullptr);
    gst_buffer_unref(buf);
}

TEST(det_meta_object_count_clamped) {
    NvmmFrameMeta f;
    fill_frame(&f, NVMM_META_MAX_OBJECTS);
    f.num_objects = NVMM_META_MAX_OBJECTS + 100;

    GstBuffer *buf = gst_buffer_new();
    GstNvmmDetMeta *m = gst_buffer_add_nvmm_det_meta(buf, &f);
    ASSERT_TRUE(m != nullptr);
    ASSERT_EQ(m->num_objects, NVMM_META_MAX_OBJECTS);
    gst_buffer_unref(buf);
}

TEST(det_meta_survives_buffer_copy) {
    NvmmFrameMeta f;
    fill_frame(&f, 5);
    GstBuffer *buf = gst_buffer_new();
    gst_buffer_add_nvmm_det_meta(buf, &f);

    GstBuffer *copy = gst_buffer_copy(buf);
    GstNvmmDetMeta *got = gst_buffer_get_nvmm_det_meta(copy);
    ASSERT_TRUE(got != nullptr);
    ASSERT_EQ(got->num_objects, 5u);
    ASSERT_EQ(got->objects[4].class_id, 4);
    ASSERT_TRUE(strcmp(got->objects[3].label, "class_3") == 0);

    gst_buffer_unref(buf);
    gst_buffer_unref(copy);
}

void fill_track(GstNvmmTrackMeta *m) {
    m->frame_number = 42;
    m->frame_width = 1920;
    m->frame_height = 1080;
    m->valid = TRUE;
    m->target_id = 7;
    m->left = 100.f; m->top = 200.f; m->width = 14.f; m->height = 8.f;
    m->object_score = 12.5f;
    m->kf_left = 101.f; m->kf_top = 201.f; m->kf_width = 14.f; m->kf_height = 8.f;
    m->kf_score = 0.8f;
    m->is_kf_only = FALSE;
    m->stable_frames = 11;
}

TEST(track_meta_add_get_roundtrip) {
    GstBuffer *buf = gst_buffer_new();
    GstNvmmTrackMeta *m = gst_buffer_add_nvmm_track_meta(buf);
    ASSERT_TRUE(m != nullptr);
    fill_track(m);

    GstNvmmTrackMeta *got = gst_buffer_get_nvmm_track_meta(buf);
    ASSERT_TRUE(got != nullptr);
    ASSERT_EQ(got->frame_number, 42u);
    ASSERT_EQ(got->frame_width, 1920u);
    ASSERT_EQ(got->target_id, 7u);
    ASSERT_TRUE(got->valid);
    ASSERT_TRUE(got->width == 14.f && got->height == 8.f);
    ASSERT_NEAR(got->object_score, 12.5f, 1e-6f);
    ASSERT_EQ(got->stable_frames, 11u);
    ASSERT_TRUE(!got->is_kf_only);
    gst_buffer_unref(buf);
}

TEST(track_meta_zero_initialized_on_add) {
    GstBuffer *buf = gst_buffer_new();
    GstNvmmTrackMeta *m = gst_buffer_add_nvmm_track_meta(buf);
    ASSERT_TRUE(m != nullptr);
    ASSERT_TRUE(!m->valid);
    ASSERT_EQ(m->target_id, 0u);
    ASSERT_TRUE(m->left == 0.f && m->width == 0.f);
    ASSERT_EQ(m->stable_frames, 0u);
    gst_buffer_unref(buf);
}

/// nvmmfusekf relies on this to overwrite the meta in place.
TEST(track_meta_fetch_or_create_is_idempotent) {
    GstBuffer *buf = gst_buffer_new();
    GstNvmmTrackMeta *a = gst_buffer_add_nvmm_track_meta(buf);
    a->target_id = 99;
    GstNvmmTrackMeta *b = gst_buffer_add_nvmm_track_meta(buf);
    ASSERT_TRUE(a == b);
    ASSERT_EQ(b->target_id, 99u);
    gst_buffer_unref(buf);
}

TEST(track_meta_survives_buffer_copy) {
    GstBuffer *buf = gst_buffer_new();
    fill_track(gst_buffer_add_nvmm_track_meta(buf));

    GstBuffer *copy = gst_buffer_copy(buf);
    GstNvmmTrackMeta *got = gst_buffer_get_nvmm_track_meta(copy);
    ASSERT_TRUE(got != nullptr);
    ASSERT_EQ(got->frame_number, 42u);
    ASSERT_EQ(got->target_id, 7u);
    ASSERT_TRUE(got->valid);
    ASSERT_TRUE(got->width == 14.f);
    ASSERT_EQ(got->stable_frames, 11u);

    gst_buffer_unref(buf);
    gst_buffer_unref(copy);
}

NvmmClassEntry class_entry(gint32 id, gfloat conf, guint32 fresh, const char *label) {
    NvmmClassEntry e{};
    e.class_id = id;
    e.confidence = conf;
    e.fresh = fresh;
    snprintf(e.label, sizeof e.label, "%s", label);
    return e;
}

TEST(class_meta_attach_and_read_back) {
    GstBuffer *buf = gst_buffer_new();
    NvmmClassEntry in[2] = { class_entry(3, 0.9f, 1, "sitting"), class_entry(-1, 0.f, 0, "") };
    ASSERT_NOT_NULL(gst_buffer_add_nvmm_class_meta(buf, in, 2));

    GstNvmmClassMeta *m = gst_buffer_get_nvmm_class_meta(buf);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->num_objects, 2u);
    ASSERT_EQ(m->objects[0].class_id, 3);
    ASSERT_NEAR(m->objects[0].confidence, 0.9f, 1e-6);
    ASSERT_EQ(m->objects[0].fresh, 1u);
    ASSERT_TRUE(strcmp(m->objects[0].label, "sitting") == 0);
    ASSERT_EQ(m->objects[1].class_id, -1);
    gst_buffer_unref(buf);
}

TEST(class_meta_empty_has_null_objects) {
    GstBuffer *buf = gst_buffer_new();
    GstNvmmClassMeta *m = gst_buffer_add_nvmm_class_meta(buf, nullptr, 0);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->num_objects, 0u);
    ASSERT_TRUE(m->objects == nullptr);
    gst_buffer_unref(buf);
}

TEST(class_meta_copy_owns_its_entries) {
    GstBuffer *buf = gst_buffer_new();
    NvmmClassEntry in[1] = { class_entry(7, 0.42f, 0, "walking") };
    gst_buffer_add_nvmm_class_meta(buf, in, 1);

    GstBuffer *copy = gst_buffer_copy(buf);
    gst_buffer_unref(buf);

    GstNvmmClassMeta *m = gst_buffer_get_nvmm_class_meta(copy);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->num_objects, 1u);
    ASSERT_EQ(m->objects[0].class_id, 7);
    ASSERT_TRUE(strcmp(m->objects[0].label, "walking") == 0);
    gst_buffer_unref(copy);
}

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    const int rc = tests_failed > 0 ? 1 : 0;
    if (exit_hangs_in_nvbufsurftransform_fini) {
        std::fflush(nullptr);
        std::_Exit(rc);
    }
    return rc;
}
