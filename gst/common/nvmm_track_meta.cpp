/// Implementation of GstNvmmTrackMeta (see nvmm_track_meta.h).

#include "nvmm_track_meta.h"

#include "nvmm_meta_util.h"

#include <cstring>

static gboolean
nvmm_track_meta_init(GstMeta *meta, gpointer /*params*/, GstBuffer * /*buffer*/)
{
    auto *m = reinterpret_cast<GstNvmmTrackMeta *>(meta);
    /* Zero everything: valid=FALSE, all coords/scores/counters 0. */
    memset(reinterpret_cast<char *>(m) + sizeof(GstMeta), 0,
           sizeof(GstNvmmTrackMeta) - sizeof(GstMeta));
    return TRUE;
}

static gboolean
nvmm_track_meta_transform(GstBuffer *dest, GstMeta *meta, GstBuffer * /*buffer*/,
                          GQuark type, gpointer /*data*/)
{
    /* The track describes frame content and survives a straight copy; any
       non-copy transform (scale/crop) must re-derive coords, so we drop it. */
    if (!GST_META_TRANSFORM_IS_COPY(type))
        return FALSE;

    auto *src = reinterpret_cast<GstNvmmTrackMeta *>(meta);
    auto *dst = reinterpret_cast<GstNvmmTrackMeta *>(
        gst_buffer_add_meta(dest, gst_nvmm_track_meta_get_info(), nullptr));
    if (!dst)
        return FALSE;
    /* POD copy of everything after the GstMeta base. */
    memcpy(reinterpret_cast<char *>(dst) + sizeof(GstMeta),
           reinterpret_cast<char *>(src) + sizeof(GstMeta),
           sizeof(GstNvmmTrackMeta) - sizeof(GstMeta));
    return TRUE;
}

GType
gst_nvmm_track_meta_api_get_type(void)
{
    static GType type = 0;
    static const gchar *tags[] = { nullptr };
    if (g_once_init_enter(&type)) {
        GType t = nvmm_meta_api_register_once("GstNvmmTrackMetaAPI", tags);
        g_once_init_leave(&type, t);
    }
    return type;
}

const GstMetaInfo *
gst_nvmm_track_meta_get_info(void)
{
    static const GstMetaInfo *info = nullptr;
    if (g_once_init_enter(&info)) {
        const GstMetaInfo *mi = gst_meta_register(
            gst_nvmm_track_meta_api_get_type(), "GstNvmmTrackMeta",
            sizeof(GstNvmmTrackMeta), nvmm_track_meta_init,
            nullptr /* no heap to free */, nvmm_track_meta_transform);
        g_once_init_leave(&info, mi);
    }
    return info;
}

GstNvmmTrackMeta *
gst_buffer_add_nvmm_track_meta(GstBuffer *buffer)
{
    g_return_val_if_fail(GST_IS_BUFFER(buffer), nullptr);
    /* Fetch-or-create: nvmmfusekf overwrites the box on the same buffer. */
    GstNvmmTrackMeta *existing = gst_buffer_get_nvmm_track_meta(buffer);
    if (existing)
        return existing;
    return reinterpret_cast<GstNvmmTrackMeta *>(
        gst_buffer_add_meta(buffer, gst_nvmm_track_meta_get_info(), nullptr));
}
