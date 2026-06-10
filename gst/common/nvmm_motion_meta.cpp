#include "nvmm_motion_meta.h"

#include "nvmm_meta_util.h"

#include <cstring>

GType
gst_nvmm_motion_meta_api_get_type(void)
{
    static GType type = 0;
    static const gchar *tags[] = { nullptr };
    if (g_once_init_enter(&type)) {
        GType t = nvmm_meta_api_register_once("GstNvmmMotionMetaAPI", tags);
        g_once_init_leave(&type, t);
    }
    return type;
}

static gboolean
nvmm_motion_meta_init(GstMeta *meta, gpointer, GstBuffer *)
{
    auto *m = reinterpret_cast<GstNvmmMotionMeta *>(meta);
    m->num_objects = 0;
    m->objects = nullptr;
    return TRUE;
}

static void
nvmm_motion_meta_free(GstMeta *meta, GstBuffer *)
{
    auto *m = reinterpret_cast<GstNvmmMotionMeta *>(meta);
    g_free(m->objects);
    m->objects = nullptr;
    m->num_objects = 0;
}

static gboolean
nvmm_motion_meta_transform(GstBuffer *dest, GstMeta *meta, GstBuffer * /*buffer*/,
                           GQuark type, gpointer /*data*/)
{
    /* Copy-transforms only; motion entries index into the det meta, which is
       likewise dropped on non-copy transforms, so the pairing stays intact. */
    if (!GST_META_TRANSFORM_IS_COPY(type))
        return FALSE;
    auto *m = reinterpret_cast<GstNvmmMotionMeta *>(meta);
    return gst_buffer_add_nvmm_motion_meta(dest, m->objects, m->num_objects) != nullptr;
}

const GstMetaInfo *
gst_nvmm_motion_meta_get_info(void)
{
    static const GstMetaInfo *info = nullptr;
    if (g_once_init_enter(&info)) {
        const GstMetaInfo *mi = gst_meta_register(
            gst_nvmm_motion_meta_api_get_type(), "GstNvmmMotionMeta",
            sizeof(GstNvmmMotionMeta), nvmm_motion_meta_init,
            nvmm_motion_meta_free, nvmm_motion_meta_transform);
        g_once_init_leave(&info, mi);
    }
    return info;
}

GstNvmmMotionMeta *
gst_buffer_add_nvmm_motion_meta(GstBuffer *buffer, const nvmm::MotionEntry *entries,
                                guint32 n)
{
    auto *m = reinterpret_cast<GstNvmmMotionMeta *>(
        gst_buffer_add_meta(buffer, gst_nvmm_motion_meta_get_info(), nullptr));
    if (!m)
        return nullptr;
    m->num_objects = n;
    if (n) {
        m->objects = static_cast<nvmm::MotionEntry *>(
            g_malloc(n * sizeof(nvmm::MotionEntry)));
        memcpy(m->objects, entries, n * sizeof(nvmm::MotionEntry));
    } else {
        m->objects = nullptr;
    }
    return m;
}
