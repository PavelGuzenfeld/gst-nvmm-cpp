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
    /* Borrowed: the element owns its ALWAYS pads for its whole lifetime, so no
       per-frame name lookup or ref churn is needed in aggregate(). */
    GstAggregatorPad *det_pad;
    GstAggregatorPad *flow_pad;
    gboolean          src_caps_set;
    gboolean          compute_motion;    /* property */
    gdouble           motion_threshold;  /* property: px/frame */
};

G_DEFINE_TYPE(GstNvmmFusion, gst_nvmm_fusion, GST_TYPE_AGGREGATOR)

enum { PROP_0, PROP_COMPUTE_MOTION, PROP_MOTION_THRESHOLD };

/* Phase-3 payoff: mean flow magnitude under each det box -> motion meta. The
   flow field is a small tightly-packed host memory (e.g. 480x270 cells). */
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
    /* Boxes are in dm->infer_width/height space; we hand the compute the flow
       meta's frame dims. These match today — both branches see the same tee'd
       frame and nvmminfer sets infer dims = source frame — but if a scaler ever
       lands between the branches the spaces diverge and need a rescale here. */
    const guint32 n = MIN(dm->num_objects, (guint32)NVMM_META_MAX_OBJECTS);
    const uint32_t got = nvmm::compute_box_motion(
        reinterpret_cast<const int16_t *>(map.data), fm->mv_width, fm->mv_height,
        fm->grid_size, fm->frame_width, fm->frame_height,
        dm->objects, n, (float)self->motion_threshold, entries);
    gst_memory_unmap(fm->mv, &map);
    if (got)
        gst_buffer_add_nvmm_motion_meta(out, entries, got);
}

/* Two named sink pads (detection carries the frame + det_meta and becomes the
   output; flow contributes its optical-flow meta) + the fused src. */
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
gst_nvmm_fusion_aggregate(GstAggregator *agg, gboolean /*timeout*/) {
    auto *self = GST_NVMM_FUSION(agg);
    GstAggregatorPad *dagg = self->det_pad;
    GstAggregatorPad *fagg = self->flow_pad;

    /* Inner join: either branch ending ends the fused stream. */
    if (gst_aggregator_pad_is_eos(dagg) || gst_aggregator_pad_is_eos(fagg))
        return GST_FLOW_EOS;

    GstBuffer *det = gst_aggregator_pad_peek_buffer(dagg);
    GstBuffer *flow = gst_aggregator_pad_peek_buffer(fagg);
    /* NEED_DATA tells the base class to wait for more input rather than spin or
       stall; only the fuse path overrides it with the finish_buffer result. */
    GstFlowReturn ret = GST_AGGREGATOR_FLOW_NEED_DATA;

    /* Wait until both branches have a frame before emitting a join. */
    if (!det || !flow)
        goto done;

    /* PTS is the join key: tee copies timestamps verbatim, so the queue heads
       normally match. If a branch dropped a frame (leaky queue, error), drop
       the OLDER head and retry — the skew self-heals instead of pairing frame
       N with frame N-1 forever. NONE timestamps pair as-is. */
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

    /* Output src caps = the detection branch's caps (the carrier frame). */
    if (!self->src_caps_set) {
        GstCaps *c = gst_pad_get_current_caps(GST_PAD(dagg));
        if (!c) goto done;  /* not negotiated yet — wait */
        gst_aggregator_set_src_caps(agg, c);
        gst_caps_unref(c);
        self->src_caps_set = TRUE;
    }

    {
        /* The detection buffer is the carrier. Read the flow meta first, then
           drop both pads' queued refs so the detection buffer is held only by
           our peek ref (refcount 1): make_writable is then a no-op, NOT a deep
           copy — copying an opaque NVMM surface via the default mem_copy would
           corrupt it (the carrier is NVIDIA NO_SHARE memory from nvvidconv). */
        NvmmOpticalFlowMeta *fm = gst_buffer_get_nvmm_optical_flow_meta(flow);
        gst_aggregator_pad_drop_buffer(dagg);
        gst_aggregator_pad_drop_buffer(fagg);

        GstBuffer *out = gst_buffer_make_writable(det);  /* consumes our peek ref */
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
        ret = gst_aggregator_finish_buffer(agg, out);  /* consumes `out` */
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

/* ALWAYS pads are not auto-created from templates — instantiate the two named
   pads here so `f.detection` / `f.flow` exist and can be linked. The class
   template carries GST_TYPE_AGGREGATOR_PAD, so new_from_template builds the
   right pad subclass. */
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
