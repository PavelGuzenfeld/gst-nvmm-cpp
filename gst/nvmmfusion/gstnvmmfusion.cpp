#include "config.h"

#include "gstnvmmfusion.h"
#include "nvmm_det_meta.h"
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
};

G_DEFINE_TYPE(GstNvmmFusion, gst_nvmm_fusion, GST_TYPE_AGGREGATOR)

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
gst_nvmm_fusion_class_init(GstNvmmFusionClass *klass) {
    auto *el = GST_ELEMENT_CLASS(klass);
    auto *agg = GST_AGGREGATOR_CLASS(klass);

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
