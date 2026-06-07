#include "config.h"  // PACKAGE_VERSION

#include "gstnvmmcompositor.h"

#include "gstnvmmallocator.h"
#include "nvmm_caps.h"

#ifdef NVMM_MOCK_API
#include "nvbufsurface_mock.h"
#else
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#endif

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_compositor_debug);
#define GST_CAT_DEFAULT gst_nvmm_compositor_debug

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

/* ===================== Request sink pad ===================== */

struct _GstNvmmCompositorPad {
    GstAggregatorPad parent;
    gint xpos, ypos, width, height;  /* placement in the output; 0 w/h = fill */
};

G_DEFINE_TYPE(GstNvmmCompositorPad, gst_nvmm_compositor_pad, GST_TYPE_AGGREGATOR_PAD)

enum { PAD_PROP_0, PAD_PROP_XPOS, PAD_PROP_YPOS, PAD_PROP_WIDTH, PAD_PROP_HEIGHT };

static void
gst_nvmm_compositor_pad_set_property(GObject* o, guint id, const GValue* v, GParamSpec* p)
{
    auto* pad = GST_NVMM_COMPOSITOR_PAD(o);
    switch (id) {
        case PAD_PROP_XPOS:   pad->xpos = g_value_get_int(v); break;
        case PAD_PROP_YPOS:   pad->ypos = g_value_get_int(v); break;
        case PAD_PROP_WIDTH:  pad->width = g_value_get_int(v); break;
        case PAD_PROP_HEIGHT: pad->height = g_value_get_int(v); break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p); break;
    }
}

static void
gst_nvmm_compositor_pad_get_property(GObject* o, guint id, GValue* v, GParamSpec* p)
{
    auto* pad = GST_NVMM_COMPOSITOR_PAD(o);
    switch (id) {
        case PAD_PROP_XPOS:   g_value_set_int(v, pad->xpos); break;
        case PAD_PROP_YPOS:   g_value_set_int(v, pad->ypos); break;
        case PAD_PROP_WIDTH:  g_value_set_int(v, pad->width); break;
        case PAD_PROP_HEIGHT: g_value_set_int(v, pad->height); break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p); break;
    }
}

static void
gst_nvmm_compositor_pad_class_init(GstNvmmCompositorPadClass* klass)
{
    auto* go = G_OBJECT_CLASS(klass);
    go->set_property = gst_nvmm_compositor_pad_set_property;
    go->get_property = gst_nvmm_compositor_pad_get_property;
    auto flags = static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS);
    g_object_class_install_property(go, PAD_PROP_XPOS,
        g_param_spec_int("xpos", "X", "Left offset in output", 0, 8192, 0, flags));
    g_object_class_install_property(go, PAD_PROP_YPOS,
        g_param_spec_int("ypos", "Y", "Top offset in output", 0, 8192, 0, flags));
    g_object_class_install_property(go, PAD_PROP_WIDTH,
        g_param_spec_int("width", "Width", "Tile width (0 = to output edge)", 0, 8192, 0, flags));
    g_object_class_install_property(go, PAD_PROP_HEIGHT,
        g_param_spec_int("height", "Height", "Tile height (0 = to output edge)", 0, 8192, 0, flags));
}

static void
gst_nvmm_compositor_pad_init(GstNvmmCompositorPad* pad)
{
    pad->xpos = pad->ypos = pad->width = pad->height = 0;
}

/* ===================== Compositor ===================== */

struct _GstNvmmCompositor {
    GstAggregator parent;
    gint out_width, out_height;
    GstAllocator* allocator;
    gboolean src_caps_set;
};

G_DEFINE_TYPE(GstNvmmCompositor, gst_nvmm_compositor, GST_TYPE_AGGREGATOR)

enum { PROP_0, PROP_WIDTH, PROP_HEIGHT };

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string){ NV12, RGBA }, "
                    "width=(int)[1,8192], height=(int)[1,8192], "
                    "framerate=(fraction)[0/1, 240/1]"));

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink_%u", GST_PAD_SINK, GST_PAD_REQUEST,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string){ NV12, RGBA }, "
                    "width=(int)[1,8192], height=(int)[1,8192], "
                    "framerate=(fraction)[0/1, 240/1]"));

static NvBufSurface*
get_surface(GstBuffer* buf)
{
    GstMemory* mem = gst_buffer_peek_memory(buf, 0);
    if (gst_is_nvmm_memory(mem))
        return static_cast<NvBufSurface*>(gst_nvmm_memory_get_surface(mem));
    GstMapInfo map;
    if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
        auto* s = reinterpret_cast<NvBufSurface*>(map.data);
        gst_buffer_unmap(buf, &map);
        return s;
    }
    return nullptr;
}

static GstFlowReturn
gst_nvmm_compositor_aggregate(GstAggregator* agg, gboolean timeout)
{
    auto* self = GST_NVMM_COMPOSITOR(agg);
    (void)timeout;

    /* Negotiate src caps once, from the configured output size. */
    if (!self->src_caps_set) {
        GstCaps* caps = gst_caps_from_string(
            "video/x-raw(memory:NVMM), format=(string)NV12");
        gst_caps_set_simple(caps, "width", G_TYPE_INT, self->out_width,
                            "height", G_TYPE_INT, self->out_height,
                            "framerate", GST_TYPE_FRACTION, 30, 1, NULL);
        gst_aggregator_set_src_caps(agg, caps);
        gst_caps_unref(caps);
        self->src_caps_set = TRUE;
    }

    /* Allocate the output NVMM frame. Each pad writes only its dst_rect via
       VIC, so regions no pad covers are left at the allocator's initial
       contents (undefined). The element therefore assumes the request pads
       tile the full output (mosaic) — for partial layouts, add a full-frame
       background pad behind the others. */
    GstMemory* omem = gst_nvmm_allocator_alloc_video(
        self->allocator, GST_VIDEO_FORMAT_NV12, self->out_width, self->out_height);
    if (!omem) {
        GST_ERROR_OBJECT(self, "Failed to allocate output NVMM buffer");
        return GST_FLOW_ERROR;
    }
    GstBuffer* outbuf = gst_buffer_new();
    gst_buffer_append_memory(outbuf, omem);

    NvBufSurface* dst = get_surface(outbuf);
    if (!dst) {
        gst_buffer_unref(outbuf);
        return GST_FLOW_ERROR;
    }
    dst->numFilled = dst->batchSize ? dst->batchSize : 1;

    gboolean any = FALSE, all_eos = TRUE;
    GstClockTime pts = GST_CLOCK_TIME_NONE;

    GST_OBJECT_LOCK(self);
    for (GList* l = GST_ELEMENT(self)->sinkpads; l; l = l->next) {
        auto* aggpad = GST_AGGREGATOR_PAD(l->data);
        auto* cpad = GST_NVMM_COMPOSITOR_PAD(l->data);
        GstBuffer* inbuf = gst_aggregator_pad_peek_buffer(aggpad);
        if (!inbuf) {
            if (!gst_aggregator_pad_is_eos(aggpad)) all_eos = FALSE;
            continue;
        }
        all_eos = FALSE;
        NvBufSurface* src = get_surface(inbuf);
        if (src) {
            src->numFilled = src->batchSize ? src->batchSize : 1;
            NvBufSurfTransformRect dr;
            dr.left   = (uint32_t)cpad->xpos;
            dr.top    = (uint32_t)cpad->ypos;
            dr.width  = (uint32_t)(cpad->width  > 0 ? cpad->width  : self->out_width  - cpad->xpos);
            dr.height = (uint32_t)(cpad->height > 0 ? cpad->height : self->out_height - cpad->ypos);
            NvBufSurfTransformParams xf;
            memset(&xf, 0, sizeof(xf));
            xf.dst_rect = &dr;
            xf.transform_flag = NVBUFSURF_TRANSFORM_CROP_DST;
            if (NvBufSurfTransform(src, dst, &xf) != NvBufSurfTransformError_Success)
                GST_WARNING_OBJECT(self, "composite transform failed for a pad");
            any = TRUE;
            if (pts == GST_CLOCK_TIME_NONE)
                pts = GST_BUFFER_PTS(inbuf);
        }
        gst_buffer_unref(inbuf);
        gst_aggregator_pad_drop_buffer(aggpad);
    }
    GST_OBJECT_UNLOCK(self);

    if (all_eos) {
        gst_buffer_unref(outbuf);
        return GST_FLOW_EOS;
    }
    if (!any) {
        gst_buffer_unref(outbuf);
        return GST_FLOW_OK;  /* nothing ready yet */
    }

    GST_BUFFER_PTS(outbuf) = pts;
    return gst_aggregator_finish_buffer(agg, outbuf);
}

static void
gst_nvmm_compositor_set_property(GObject* o, guint id, const GValue* v, GParamSpec* p)
{
    auto* self = GST_NVMM_COMPOSITOR(o);
    switch (id) {
        case PROP_WIDTH:  self->out_width = g_value_get_int(v); break;
        case PROP_HEIGHT: self->out_height = g_value_get_int(v); break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p); break;
    }
}

static void
gst_nvmm_compositor_get_property(GObject* o, guint id, GValue* v, GParamSpec* p)
{
    auto* self = GST_NVMM_COMPOSITOR(o);
    switch (id) {
        case PROP_WIDTH:  g_value_set_int(v, self->out_width); break;
        case PROP_HEIGHT: g_value_set_int(v, self->out_height); break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p); break;
    }
}

static void
gst_nvmm_compositor_finalize(GObject* o)
{
    auto* self = GST_NVMM_COMPOSITOR(o);
    if (self->allocator) gst_object_unref(self->allocator);
    G_OBJECT_CLASS(gst_nvmm_compositor_parent_class)->finalize(o);
}

static void
gst_nvmm_compositor_class_init(GstNvmmCompositorClass* klass)
{
    auto* go = G_OBJECT_CLASS(klass);
    auto* el = GST_ELEMENT_CLASS(klass);
    auto* agg = GST_AGGREGATOR_CLASS(klass);

    go->set_property = gst_nvmm_compositor_set_property;
    go->get_property = gst_nvmm_compositor_get_property;
    go->finalize = gst_nvmm_compositor_finalize;
    agg->aggregate = gst_nvmm_compositor_aggregate;

    auto flags = static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS);
    g_object_class_install_property(go, PROP_WIDTH,
        g_param_spec_int("width", "Output width", "Output frame width",
                         1, 8192, 1280, flags));
    g_object_class_install_property(go, PROP_HEIGHT,
        g_param_spec_int("height", "Output height", "Output frame height",
                         1, 8192, 720, flags));

    gst_element_class_add_static_pad_template_with_gtype(
        el, &src_template, GST_TYPE_AGGREGATOR_PAD);
    gst_element_class_add_static_pad_template_with_gtype(
        el, &sink_template, GST_TYPE_NVMM_COMPOSITOR_PAD);

    gst_element_class_set_static_metadata(el,
        "NVMM Compositor", "Filter/Editor/Video/Compositor",
        "VIC-composited multi-input NVMM mixer (mosaic/overlay)",
        "Pavel Guzenfeld, Stereolabs");

    GST_DEBUG_CATEGORY_INIT(gst_nvmm_compositor_debug, "nvmmcompositor", 0,
                            "NVMM compositor");
}

static void
gst_nvmm_compositor_init(GstNvmmCompositor* self)
{
    self->out_width = 1280;
    self->out_height = 720;
    self->allocator = gst_nvmm_allocator_new(0);
    self->src_caps_set = FALSE;
}

/* ===================== Plugin ===================== */

static gboolean
plugin_init(GstPlugin* plugin)
{
    return gst_element_register(plugin, "nvmmcompositor",
                               GST_RANK_NONE, GST_TYPE_NVMM_COMPOSITOR);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    nvmmcompositor, "NVMM VIC compositor",
    plugin_init, PACKAGE_VERSION, "LGPL", "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp")
