#include "config.h"

#include "gstnvmmfusion.h"
#include "nvmm_det_meta.h"
#include "nvmm_motion.hpp"
#include "nvmm_motion_meta.h"
#include "nvmm_optical_flow_meta.h"

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_fusion_debug);
#define GST_CAT_DEFAULT gst_nvmm_fusion_debug

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

struct _GstNvmmFusion {
    GstAggregator parent;
    GstAggregatorPad *det_pad;
    GstAggregatorPad *flow_pad;
    gboolean          src_caps_set;
    gboolean          compute_motion;
    gdouble           motion_threshold;
};

G_DEFINE_TYPE(GstNvmmFusion, gst_nvmm_fusion, GST_TYPE_AGGREGATOR)

enum { PROP_0, PROP_COMPUTE_MOTION, PROP_MOTION_THRESHOLD };

static void
annotate_motion(GstNvmmFusion *self, GstBuffer *out,
                const GstNvmmDetMeta *dm, const NvmmOpticalFlowMeta *fm)
{
    GstMapInfo map;
    if (!fm->mv || !gst_memory_map(fm->mv, &map, GST_MAP_READ)) {
        GST_WARNING_OBJECT(self, "cannot map flow field; skipping motion");
        return;
    }
    nvmm::MotionEntry entries[NVMM_META_MAX_OBJECTS];
    const guint32 n = MIN(dm->num_objects, (guint32)NVMM_META_MAX_OBJECTS);
    const uint32_t got = nvmm::compute_box_motion(
        reinterpret_cast<const int16_t *>(map.data), fm->mv_width, fm->mv_height,
        fm->grid_size, fm->frame_width, fm->frame_height,
        dm->objects, n, (float)self->motion_threshold, entries);
    gst_memory_unmap(fm->mv, &map);
    if (got)
        gst_buffer_add_nvmm_motion_meta(out, entries, got);
}

static GstStaticPadTemplate det_tmpl = GST_STATIC_PAD_TEMPLATE(
    "detection", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string)NV12"));
static GstStaticPadTemplate flow_tmpl = GST_STATIC_PAD_TEMPLATE(
    "flow", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string)NV12"));
static GstStaticPadTemplate src_tmpl = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string)NV12"));

static GstFlowReturn
gst_nvmm_fusion_aggregate(GstAggregator *agg, gboolean ) {
    auto *self = GST_NVMM_FUSION(agg);
    GstAggregatorPad *dagg = self->det_pad;
    GstAggregatorPad *fagg = self->flow_pad;

    if (gst_aggregator_pad_is_eos(dagg) || gst_aggregator_pad_is_eos(fagg))
        return GST_FLOW_EOS;

    GstBuffer *det = gst_aggregator_pad_peek_buffer(dagg);
    GstBuffer *flow = gst_aggregator_pad_peek_buffer(fagg);
    GstFlowReturn ret = GST_AGGREGATOR_FLOW_NEED_DATA;

    if (!det || !flow)
        goto done;

    {
        const GstClockTime dp = GST_BUFFER_PTS(det), fp = GST_BUFFER_PTS(flow);
        if (GST_CLOCK_TIME_IS_VALID(dp) && GST_CLOCK_TIME_IS_VALID(fp) && dp != fp) {
            GST_WARNING_OBJECT(self,
                "PTS mismatch det=%" GST_TIME_FORMAT " flow=%" GST_TIME_FORMAT
                " — dropping the older head to resync",
                GST_TIME_ARGS(dp), GST_TIME_ARGS(fp));
            gst_aggregator_pad_drop_buffer(dp < fp ? dagg : fagg);
            goto done;
        }
    }

    if (!self->src_caps_set) {
        GstCaps *c = gst_pad_get_current_caps(GST_PAD(dagg));
        if (!c) goto done;
        gst_aggregator_set_src_caps(agg, c);
        gst_caps_unref(c);
        self->src_caps_set = TRUE;
    }

    {
        NvmmOpticalFlowMeta *fm = gst_buffer_get_nvmm_optical_flow_meta(flow);
        gst_aggregator_pad_drop_buffer(dagg);
        gst_aggregator_pad_drop_buffer(fagg);

        GstBuffer *out = gst_buffer_make_writable(det);
        det = nullptr;
        if (fm) {
            gst_buffer_add_nvmm_optical_flow_meta(out, fm->mv, fm->mv_width,
                fm->mv_height, fm->grid_size, fm->frame_width, fm->frame_height);
        } else {
            GST_LOG_OBJECT(self, "flow branch buffer carries no optical-flow meta");
        }
        GstNvmmDetMeta *dm = gst_buffer_get_nvmm_det_meta(out);
        if (self->compute_motion && fm && dm && dm->num_objects)
            annotate_motion(self, out, dm, fm);
        GST_LOG_OBJECT(self, "fused: %u detection(s) + flow=%s on one buffer",
                       dm ? dm->num_objects : 0, fm ? "yes" : "no");
        ret = gst_aggregator_finish_buffer(agg, out);
    }

done:
    if (det) gst_buffer_unref(det);
    if (flow) gst_buffer_unref(flow);
    return ret;
}

static void
gst_nvmm_fusion_set_property(GObject *o, guint id, const GValue *v, GParamSpec *p) {
    auto *self = GST_NVMM_FUSION(o);
    if (id == PROP_COMPUTE_MOTION) self->compute_motion = g_value_get_boolean(v);
    else if (id == PROP_MOTION_THRESHOLD) self->motion_threshold = g_value_get_double(v);
    else G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p);
}

static void
gst_nvmm_fusion_get_property(GObject *o, guint id, GValue *v, GParamSpec *p) {
    auto *self = GST_NVMM_FUSION(o);
    if (id == PROP_COMPUTE_MOTION) g_value_set_boolean(v, self->compute_motion);
    else if (id == PROP_MOTION_THRESHOLD) g_value_set_double(v, self->motion_threshold);
    else G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p);
}

static void
gst_nvmm_fusion_class_init(GstNvmmFusionClass *klass) {
    auto *go = G_OBJECT_CLASS(klass);
    auto *el = GST_ELEMENT_CLASS(klass);
    auto *agg = GST_AGGREGATOR_CLASS(klass);

    go->set_property = gst_nvmm_fusion_set_property;
    go->get_property = gst_nvmm_fusion_get_property;
    g_object_class_install_property(go, PROP_COMPUTE_MOTION,
        g_param_spec_boolean("compute-motion", "Compute motion",
            "Compute per-detection motion from the flow field and attach a "
            "GstNvmmMotionMeta (the Phase-3 fusion result)", TRUE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(go, PROP_MOTION_THRESHOLD,
        g_param_spec_double("motion-threshold", "Motion threshold",
            "Mean flow magnitude (pixels/frame) at/above which an object is "
            "flagged moving", 0.0, 1000.0, 1.0,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_add_static_pad_template_with_gtype(
        el, &det_tmpl, GST_TYPE_AGGREGATOR_PAD);
    gst_element_class_add_static_pad_template_with_gtype(
        el, &flow_tmpl, GST_TYPE_AGGREGATOR_PAD);
    gst_element_class_add_static_pad_template_with_gtype(
        el, &src_tmpl, GST_TYPE_AGGREGATOR_PAD);
    gst_element_class_set_static_metadata(el,
        "NVMM Branch Fusion", "Filter/Aggregator/Video",
        "Join the detector and optical-flow branches by PTS, unioning their "
        "GstNvmmDetMeta + NvmmOpticalFlowMeta onto one buffer (no DeepStream)",
        "Pavel Guzenfeld");

    agg->aggregate = gst_nvmm_fusion_aggregate;

    GST_DEBUG_CATEGORY_INIT(gst_nvmm_fusion_debug, "nvmmfusion", 0,
                            "NVMM branch fusion");
}

static GstAggregatorPad *
add_sink_pad(GstNvmmFusion *self, const char *name) {
    GstPadTemplate *tmpl =
        gst_element_class_get_pad_template(GST_ELEMENT_GET_CLASS(self), name);
    GstPad *pad = gst_pad_new_from_template(tmpl, name);
    gst_element_add_pad(GST_ELEMENT(self), pad);
    return GST_AGGREGATOR_PAD(pad);
}

static void
gst_nvmm_fusion_init(GstNvmmFusion *self) {
    self->src_caps_set = FALSE;
    self->compute_motion = TRUE;
    self->motion_threshold = 1.0;
    self->det_pad = add_sink_pad(self, "detection");
    self->flow_pad = add_sink_pad(self, "flow");
}

static gboolean plugin_init(GstPlugin *plugin) {
    return gst_element_register(plugin, "nvmmfusion", GST_RANK_NONE,
                                GST_TYPE_NVMM_FUSION);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    nvmmfusion, "Join detector + optical-flow branches by PTS, union their metas",
    plugin_init, PACKAGE_VERSION, "LGPL", "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp")
