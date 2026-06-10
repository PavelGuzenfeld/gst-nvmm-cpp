#include "nvmm_class_meta.h"

#include "nvmm_meta_util.h"

#include <cstring>

GType
gst_nvmm_class_meta_api_get_type(void)
{
    static GType type = 0;
    static const gchar *tags[] = { nullptr };
    if (g_once_init_enter(&type)) {
        GType t = nvmm_meta_api_register_once("GstNvmmClassMetaAPI", tags);
        g_once_init_leave(&type, t);
    }
    return type;
}

static gboolean
nvmm_class_meta_init(GstMeta *meta, gpointer, GstBuffer *)
{
    auto *m = reinterpret_cast<GstNvmmClassMeta *>(meta);
    m->num_objects = 0;
    m->objects = nullptr;
    return TRUE;
}

static void
nvmm_class_meta_free(GstMeta *meta, GstBuffer *)
{
    auto *m = reinterpret_cast<GstNvmmClassMeta *>(meta);
    g_free(m->objects);
    m->objects = nullptr;
    m->num_objects = 0;
}

static gboolean
nvmm_class_meta_transform(GstBuffer *dest, GstMeta *meta, GstBuffer * /*buffer*/,
                          GQuark type, gpointer /*data*/)
{
    /* Copy-transforms only; class entries index into the det meta, which is
       likewise dropped on non-copy transforms, so the pairing stays intact. */
    if (!GST_META_TRANSFORM_IS_COPY(type))
        return FALSE;
    auto *m = reinterpret_cast<GstNvmmClassMeta *>(meta);
    return gst_buffer_add_nvmm_class_meta(dest, m->objects, m->num_objects) != nullptr;
}

const GstMetaInfo *
gst_nvmm_class_meta_get_info(void)
{
    static const GstMetaInfo *info = nullptr;
    if (g_once_init_enter(&info)) {
        const GstMetaInfo *mi = gst_meta_register(
            gst_nvmm_class_meta_api_get_type(), "GstNvmmClassMeta",
            sizeof(GstNvmmClassMeta), nvmm_class_meta_init,
            nvmm_class_meta_free, nvmm_class_meta_transform);
        g_once_init_leave(&info, mi);
    }
    return info;
}

GstNvmmClassMeta *
gst_buffer_add_nvmm_class_meta(GstBuffer *buffer, const NvmmClassEntry *entries,
                               guint32 n)
{
    auto *m = reinterpret_cast<GstNvmmClassMeta *>(
        gst_buffer_add_meta(buffer, gst_nvmm_class_meta_get_info(), nullptr));
    if (!m)
        return nullptr;
    m->num_objects = n;
    if (n) {
        m->objects = static_cast<NvmmClassEntry *>(
            g_malloc(n * sizeof(NvmmClassEntry)));
        memcpy(m->objects, entries, n * sizeof(NvmmClassEntry));
    } else {
        m->objects = nullptr;
    }
    return m;
}
