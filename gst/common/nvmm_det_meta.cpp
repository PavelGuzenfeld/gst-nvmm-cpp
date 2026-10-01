#include "nvmm_det_meta.h"

#include "nvmm_meta_util.h"

#include <cstring>

#ifdef NVMM_DEEPSTREAM_META
#include <nvdsmeta.h>
#endif

static gboolean
nvmm_det_meta_init(GstMeta *meta, gpointer, GstBuffer *)
{
    auto *m = reinterpret_cast<GstNvmmDetMeta *>(meta);
    m->frame_number = 0;
    m->infer_width = 0;
    m->infer_height = 0;
    m->flags = 0;
    m->num_objects = 0;
    m->objects = nullptr;
    return TRUE;
}

static void
nvmm_det_meta_free(GstMeta *meta, GstBuffer *)
{
    auto *m = reinterpret_cast<GstNvmmDetMeta *>(meta);
    g_free(m->objects);
    m->objects = nullptr;
    m->num_objects = 0;
}

/// `n` must already be clamped to NVMM_META_MAX_OBJECTS.
static GstNvmmDetMeta *
nvmm_det_meta_attach(GstBuffer *buffer, guint64 frame_number, guint32 infer_width,
                     guint32 infer_height, guint32 flags, guint n,
                     const NvmmDetObject *objects)
{
    auto *m = reinterpret_cast<GstNvmmDetMeta *>(
        gst_buffer_add_meta(buffer, gst_nvmm_det_meta_get_info(), nullptr));
    if (!m)
        return nullptr;

    m->frame_number = frame_number;
    m->infer_width = infer_width;
    m->infer_height = infer_height;
    m->flags = flags;
    m->num_objects = n;
    if (n) {
        m->objects = static_cast<NvmmDetObject *>(g_malloc(n * sizeof(NvmmDetObject)));
        memcpy(m->objects, objects, n * sizeof(NvmmDetObject));
    } else {
        m->objects = nullptr;
    }
    return m;
}

/// Scale/crop would need the boxes re-derived, so only a straight copy
/// keeps the meta.
static gboolean
nvmm_det_meta_transform(GstBuffer *dest, GstMeta *meta, GstBuffer *,
                        GQuark type, gpointer)
{
    if (!GST_META_TRANSFORM_IS_COPY(type))
        return FALSE;

    auto *src = reinterpret_cast<GstNvmmDetMeta *>(meta);
    return nvmm_det_meta_attach(dest, src->frame_number, src->infer_width,
                                src->infer_height, src->flags, src->num_objects,
                                src->objects) != nullptr;
}

GType
gst_nvmm_det_meta_api_get_type(void)
{
    static GType type = 0;
    static const gchar *tags[] = { nullptr };
    if (g_once_init_enter(&type)) {
        GType t = nvmm_meta_api_register_once("GstNvmmDetMetaAPI", tags);
        g_once_init_leave(&type, t);
    }
    return type;
}

const GstMetaInfo *
gst_nvmm_det_meta_get_info(void)
{
    static const GstMetaInfo *info = nullptr;
    if (g_once_init_enter(&info)) {
        const GstMetaInfo *mi = gst_meta_register(
            gst_nvmm_det_meta_api_get_type(), "GstNvmmDetMeta",
            sizeof(GstNvmmDetMeta), nvmm_det_meta_init, nvmm_det_meta_free,
            nvmm_det_meta_transform);
        g_once_init_leave(&info, mi);
    }
    return info;
}

GstNvmmDetMeta *
gst_buffer_add_nvmm_det_meta(GstBuffer *buffer, const NvmmFrameMeta *frame)
{
    g_return_val_if_fail(GST_IS_BUFFER(buffer), nullptr);
    g_return_val_if_fail(frame != nullptr, nullptr);

    guint n = frame->num_objects;
    if (n > NVMM_META_MAX_OBJECTS)
        n = NVMM_META_MAX_OBJECTS;

    return nvmm_det_meta_attach(buffer, frame->frame_number, frame->infer_width,
                                frame->infer_height, frame->flags, n,
                                frame->objects);
}

#ifdef NVMM_DEEPSTREAM_META
/// DeepStream marks untracked objects all-Fs; the wire contract uses 0.
guint
nvmm_frame_meta_from_nvds(void *batch, guint frame_index,
                          guint32 infer_w, guint32 infer_h,
                          guint64 frame_number, NvmmFrameMeta *out)
{
    out->frame_number = frame_number;
    out->infer_width = infer_w;
    out->infer_height = infer_h;
    out->num_objects = 0;
    out->flags = 0;

    auto *bmeta = static_cast<NvDsBatchMeta *>(batch);
    if (!bmeta)
        return 0;

    NvDsFrameMeta *frame = nullptr;
    for (NvDsMetaList *l = bmeta->frame_meta_list; l; l = l->next) {
        auto *fm = static_cast<NvDsFrameMeta *>(l->data);
        if (fm && fm->batch_id == frame_index) { frame = fm; break; }
    }
    if (!frame)
        return 0;

    guint n = 0;
    for (NvDsMetaList *l = frame->obj_meta_list; l; l = l->next) {
        if (n >= NVMM_META_MAX_OBJECTS) {
            out->flags |= NVMM_FRAME_META_FLAG_TRUNCATED;
            break;
        }
        auto *obj = static_cast<NvDsObjectMeta *>(l->data);
        if (!obj)
            continue;
        NvmmDetObject *d = &out->objects[n];
        d->left = obj->rect_params.left;
        d->top = obj->rect_params.top;
        d->width = obj->rect_params.width;
        d->height = obj->rect_params.height;
        d->class_id = obj->class_id;
        d->confidence = (float)obj->confidence;
        d->tracker_id = (obj->object_id == 0xFFFFFFFFFFFFFFFFULL)
                            ? 0u : (uint64_t)obj->object_id;
        const char *lbl = obj->obj_label;
        if (lbl) {
            g_strlcpy(d->label, lbl, NVMM_META_LABEL_LEN);
        } else {
            d->label[0] = '\0';
        }
        n++;
    }
    out->num_objects = n;
    return n;
}
#endif
