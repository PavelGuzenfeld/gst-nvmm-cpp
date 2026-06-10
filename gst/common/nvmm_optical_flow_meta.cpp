#include "nvmm_optical_flow_meta.h"

#include "nvmm_meta_util.h"

GType
nvmm_optical_flow_meta_api_get_type(void)
{
    static GType type = 0;
    static const gchar *tags[] = { "memory", NULL };
    if (g_once_init_enter(&type)) {
        GType t = nvmm_meta_api_register_once("NvmmOpticalFlowMetaAPI", tags);
        g_once_init_leave(&type, t);
    }
    return type;
}

static gboolean
nvmm_optical_flow_meta_init(GstMeta *meta, gpointer, GstBuffer *)
{
    auto *m = reinterpret_cast<NvmmOpticalFlowMeta *>(meta);
    m->mv = nullptr;
    m->mv_width = m->mv_height = m->grid_size = 0;
    m->frame_width = m->frame_height = 0;
    return TRUE;
}

static void
nvmm_optical_flow_meta_free(GstMeta *meta, GstBuffer *)
{
    auto *m = reinterpret_cast<NvmmOpticalFlowMeta *>(meta);
    if (m->mv) {
        gst_memory_unref(m->mv);
        m->mv = nullptr;
    }
}

static gboolean
nvmm_optical_flow_meta_transform(GstBuffer *dest, GstMeta *meta, GstBuffer *,
                                 GQuark type, gpointer)
{
    /* Only honor a straight copy — the flow field belongs to the frame it was
       computed for; on any transform that isn't a pure copy, drop it rather
       than mis-associate vectors with a changed frame. */
    if (!GST_META_TRANSFORM_IS_COPY(type))
        return FALSE;

    auto *src = reinterpret_cast<NvmmOpticalFlowMeta *>(meta);
    NvmmOpticalFlowMeta *dst = gst_buffer_add_nvmm_optical_flow_meta(
        dest, src->mv, src->mv_width, src->mv_height, src->grid_size,
        src->frame_width, src->frame_height);
    return dst != nullptr;
}

const GstMetaInfo *
nvmm_optical_flow_meta_get_info(void)
{
    static const GstMetaInfo *info = nullptr;
    if (g_once_init_enter(&info)) {
        const GstMetaInfo *mi = gst_meta_register(
            NVMM_OPTICAL_FLOW_META_API_TYPE, "NvmmOpticalFlowMeta",
            sizeof(NvmmOpticalFlowMeta),
            nvmm_optical_flow_meta_init,
            nvmm_optical_flow_meta_free,
            nvmm_optical_flow_meta_transform);
        g_once_init_leave(&info, mi);
    }
    return info;
}

NvmmOpticalFlowMeta *
gst_buffer_add_nvmm_optical_flow_meta(GstBuffer *buffer, GstMemory *mv,
                                      gint mv_width, gint mv_height, gint grid_size,
                                      gint frame_width, gint frame_height)
{
    g_return_val_if_fail(GST_IS_BUFFER(buffer), nullptr);

    auto *m = reinterpret_cast<NvmmOpticalFlowMeta *>(
        gst_buffer_add_meta(buffer, nvmm_optical_flow_meta_get_info(), nullptr));
    if (!m) return nullptr;

    m->mv = mv ? gst_memory_ref(mv) : nullptr;
    m->mv_width = mv_width;
    m->mv_height = mv_height;
    m->grid_size = grid_size;
    m->frame_width = frame_width;
    m->frame_height = frame_height;
    return m;
}
