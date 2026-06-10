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

/* Allocate + populate a GstNvmmDetMeta on `buffer` from flat fields, deep-copying
   the `n` objects (n must already be clamped to NVMM_META_MAX_OBJECTS; `objects`
   may be null when n == 0). Shared by the public add and the copy-transform so the
   object array is copied exactly once, with no NvmmFrameMeta-sized temporary. */
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

static gboolean
nvmm_det_meta_transform(GstBuffer *dest, GstMeta *meta, GstBuffer * /*buffer*/,
                        GQuark type, gpointer /*data*/)
{
    /* Copy only on copy-transforms (e.g. gst_buffer_copy); for any non-copy
       transform (scale/crop) we drop the meta rather than remap it. Detections
       describe frame content and survive a straight copy, but callers that
       resize must re-derive coordinates. */
    if (!GST_META_TRANSFORM_IS_COPY(type))
        return FALSE;

    /* src->num_objects is already clamped (set when src was attached). */
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
        /* Defensive: nvmm_common is a shared lib in-tree (one copy/process), but
           a test harness or out-of-tree consumer may still static-link a second
           copy. gst_meta_api_type_register() is NOT idempotent, so reuse an
           existing registration; if we still lose a concurrent race, re-look-up
           the winner's type (the register call returns 0 on a duplicate name). */
        GType t = g_type_from_name("GstNvmmDetMetaAPI");
        if (t == 0)
            t = gst_meta_api_type_register("GstNvmmDetMetaAPI", tags);
        if (t == 0)
            t = g_type_from_name("GstNvmmDetMetaAPI");
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
