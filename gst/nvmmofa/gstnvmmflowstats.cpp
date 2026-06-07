#include "gstnvmmflowstats.h"
#include "nvmm_optical_flow_meta.h"

#include <cmath>
#include <cstdint>

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_flowstats_debug);
#define GST_CAT_DEFAULT gst_nvmm_flowstats_debug

struct _GstNvmmFlowStats {
    GstBaseSink parent;
    gboolean silent;        /* property: suppress per-frame lines */
    guint64 frames;         /* frames seen */
    guint64 frames_with_flow;
    double sum_mean_mag;    /* accumulated per-frame mean magnitude */
};

G_DEFINE_TYPE(GstNvmmFlowStats, gst_nvmm_flowstats, GST_TYPE_BASE_SINK)

enum { PROP_0, PROP_SILENT };

static GstStaticPadTemplate sink_tmpl = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string)NV12"));

static GstFlowReturn
gst_nvmm_flowstats_render(GstBaseSink *sink, GstBuffer *buf)
{
    auto *self = GST_NVMM_FLOWSTATS(sink);
    self->frames++;

    NvmmOpticalFlowMeta *m = gst_buffer_get_nvmm_optical_flow_meta(buf);
    if (!m || !m->mv) {
        GST_LOG_OBJECT(self, "frame %" G_GUINT64_FORMAT ": no flow meta", self->frames);
        return GST_FLOW_OK;
    }

    GstMapInfo map;
    if (!gst_memory_map(m->mv, &map, GST_MAP_READ))
        return GST_FLOW_OK;

    /* Tightly-packed mv_width*mv_height cells, two int16 (dx, dy) in S10.5. */
    const int16_t *v = reinterpret_cast<const int16_t *>(map.data);
    const gsize cells = (gsize)m->mv_width * m->mv_height;
    double sum = 0.0, maxmag = 0.0;
    for (gsize i = 0; i < cells; i++) {
        const double dx = v[2 * i]     / 32.0;  /* S10.5 -> pixels */
        const double dy = v[2 * i + 1] / 32.0;
        const double mag = std::sqrt(dx * dx + dy * dy);
        sum += mag;
        if (mag > maxmag) maxmag = mag;
    }
    gst_memory_unmap(m->mv, &map);

    const double mean = cells ? sum / cells : 0.0;
    self->frames_with_flow++;
    self->sum_mean_mag += mean;

    if (!self->silent) {
        GST_INFO_OBJECT(self,
            "frame %" G_GUINT64_FORMAT ": flow %dx%d grid=%d  mean=%.2f px  max=%.2f px",
            self->frames, m->mv_width, m->mv_height, m->grid_size, mean, maxmag);
        g_print("[nvmmflowstats] frame %" G_GUINT64_FORMAT
                ": %dx%d grid=%d mean=%.2f px max=%.2f px\n",
                self->frames, m->mv_width, m->mv_height, m->grid_size, mean, maxmag);
    }
    return GST_FLOW_OK;
}

static gboolean
gst_nvmm_flowstats_stop(GstBaseSink *sink)
{
    auto *self = GST_NVMM_FLOWSTATS(sink);
    const double avg = self->frames_with_flow
                       ? self->sum_mean_mag / self->frames_with_flow : 0.0;
    g_print("[nvmmflowstats] summary: %" G_GUINT64_FORMAT " frames, %"
            G_GUINT64_FORMAT " with flow, avg mean magnitude %.2f px\n",
            self->frames, self->frames_with_flow, avg);
    return TRUE;
}

static void
gst_nvmm_flowstats_set_property(GObject *o, guint id, const GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_FLOWSTATS(o);
    if (id == PROP_SILENT) self->silent = g_value_get_boolean(v);
    else G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p);
}

static void
gst_nvmm_flowstats_get_property(GObject *o, guint id, GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_FLOWSTATS(o);
    if (id == PROP_SILENT) g_value_set_boolean(v, self->silent);
    else G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p);
}

static void
gst_nvmm_flowstats_class_init(GstNvmmFlowStatsClass *klass)
{
    auto *go = G_OBJECT_CLASS(klass);
    auto *el = GST_ELEMENT_CLASS(klass);
    auto *bs = GST_BASE_SINK_CLASS(klass);

    go->set_property = gst_nvmm_flowstats_set_property;
    go->get_property = gst_nvmm_flowstats_get_property;
    g_object_class_install_property(go, PROP_SILENT,
        g_param_spec_boolean("silent", "Silent", "Suppress per-frame log lines",
            FALSE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_add_static_pad_template(el, &sink_tmpl);
    gst_element_class_set_static_metadata(el,
        "NVMM Optical-Flow Stats", "Sink/Analyzer/Video",
        "Reads NvmmOpticalFlowMeta and reports per-frame motion-vector magnitude",
        "Pavel Guzenfeld, Stereolabs");

    bs->render = gst_nvmm_flowstats_render;
    bs->stop = gst_nvmm_flowstats_stop;

    GST_DEBUG_CATEGORY_INIT(gst_nvmm_flowstats_debug, "nvmmflowstats", 0,
                            "NVMM optical-flow stats consumer");
}

static void
gst_nvmm_flowstats_init(GstNvmmFlowStats *self)
{
    self->silent = FALSE;
    self->frames = 0;
    self->frames_with_flow = 0;
    self->sum_mean_mag = 0.0;
}
