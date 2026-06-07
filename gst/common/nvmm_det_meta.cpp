/// Implementation of GstNvmmDetMeta (see nvmm_det_meta.h).

#include "nvmm_det_meta.h"

#include <cstring>

#ifdef NVMM_DEEPSTREAM_META
#include <nvdsmeta.h>
#endif

static gboolean
nvmm_det_meta_init(GstMeta *meta, gpointer /*params*/, GstBuffer * /*buffer*/)
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
nvmm_det_meta_free(GstMeta *meta, GstBuffer * /*buffer*/)
{
    auto *m = reinterpret_cast<GstNvmmDetMeta *>(meta);
    g_free(m->objects);
    m->objects = nullptr;
    m->num_objects = 0;
}

static gboolean
nvmm_det_meta_transform(GstBuffer *dest, GstMeta *meta, GstBuffer * /*buffer*/,
                        GQuark type, gpointer /*data*/)
{
    /* Copy on any transform (incl. gst_buffer_copy). Detections describe the
       frame content, so they survive a straight copy; we don't try to remap on
       scale/crop transforms — callers that resize must re-derive coordinates. */
    if (!GST_META_TRANSFORM_IS_COPY(type))
        return FALSE;

    auto *src = reinterpret_cast<GstNvmmDetMeta *>(meta);
    NvmmFrameMeta frame;
    frame.frame_number = src->frame_number;
    frame.infer_width = src->infer_width;
    frame.infer_height = src->infer_height;
    frame.flags = src->flags;
    frame.num_objects = src->num_objects;
    if (src->num_objects)
        memcpy(frame.objects, src->objects,
               src->num_objects * sizeof(NvmmDetObject));
    return gst_buffer_add_nvmm_det_meta(dest, &frame) != nullptr;
}

GType
gst_nvmm_det_meta_api_get_type(void)
{
    static GType type = 0;
    static const gchar *tags[] = { nullptr };
    if (g_once_init_enter(&type)) {
        /* nvmm_common is statically linked into each plugin .so (and into test
           harnesses), so more than one module can reach this in a single process.
           gst_meta_api_type_register() is NOT idempotent — it aborts if the API
           name is already registered — so reuse an existing registration when a
           sibling module got here first. */
        GType t = g_type_from_name("GstNvmmDetMetaAPI");
        if (t == 0)
            t = gst_meta_api_type_register("GstNvmmDetMetaAPI", tags);
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

    auto *m = reinterpret_cast<GstNvmmDetMeta *>(
        gst_buffer_add_meta(buffer, gst_nvmm_det_meta_get_info(), nullptr));
    if (!m)
        return nullptr;

    guint n = frame->num_objects;
    if (n > NVMM_META_MAX_OBJECTS)
        n = NVMM_META_MAX_OBJECTS;

    m->frame_number = frame->frame_number;
    m->infer_width = frame->infer_width;
    m->infer_height = frame->infer_height;
    m->flags = frame->flags;
    m->num_objects = n;
    if (n) {
        m->objects = static_cast<NvmmDetObject *>(
            g_malloc(n * sizeof(NvmmDetObject)));
        memcpy(m->objects, frame->objects, n * sizeof(NvmmDetObject));
    } else {
        m->objects = nullptr;
    }
    return m;
}

#ifdef NVMM_DEEPSTREAM_META
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

    /* Find the requested frame in the batch. */
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
        /* Normalize the no-tracker sentinel (UNTRACKED_OBJECT_ID == all-Fs) to 0
           to match the wire contract ("0 when no tracker"). */
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
