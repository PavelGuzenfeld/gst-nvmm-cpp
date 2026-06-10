#include "config.h"

#include "gstnvmmtracker.h"
#include "nvmm_det_meta.h"
#include "tracker.hpp"

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_tracker_debug);
#define GST_CAT_DEFAULT gst_nvmm_tracker_debug

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

struct _GstNvmmTracker {
    GstBaseTransform parent;
    gdouble        iou_threshold;   /* property */
    gint           max_age;         /* property */
    nvmm::Tracker *tracker;         /* created in start() from the properties */
};

G_DEFINE_TYPE(GstNvmmTracker, gst_nvmm_tracker, GST_TYPE_BASE_TRANSFORM)

enum { PROP_0, PROP_IOU_THRESHOLD, PROP_MAX_AGE };

/* Pixel-format-agnostic: the tracker only touches metadata, so it passes any
   NVMM frame through unchanged (same caps in/out). */
static GstStaticPadTemplate sink_tmpl = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM)"));
static GstStaticPadTemplate src_tmpl = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM)"));

static gboolean gst_nvmm_tracker_start(GstBaseTransform *bt) {
    auto *self = GST_NVMM_TRACKER(bt);
    nvmm::TrackerParams p;
    p.iou_threshold = (float)self->iou_threshold;
    p.max_age = self->max_age;
    self->tracker = new nvmm::Tracker(p);
    return TRUE;
}

static gboolean gst_nvmm_tracker_stop(GstBaseTransform *bt) {
    auto *self = GST_NVMM_TRACKER(bt);
    delete self->tracker;
    self->tracker = nullptr;
    return TRUE;
}

static GstFlowReturn gst_nvmm_tracker_transform_ip(GstBaseTransform *bt, GstBuffer *buf) {
    auto *self = GST_NVMM_TRACKER(bt);
    GstNvmmDetMeta *m = gst_buffer_get_nvmm_det_meta(buf);
    if (self->tracker && m && m->num_objects) {
        self->tracker->update(m->objects, m->num_objects);
        for (guint32 i = 0; i < m->num_objects; i++) {
            const NvmmDetObject &o = m->objects[i];
            GST_LOG_OBJECT(self, "  %s id=%" G_GUINT64_FORMAT " box=(%.0f,%.0f %0.fx%.0f)",
                           o.label, o.tracker_id, o.left, o.top, o.width, o.height);
        }
    }
    return GST_FLOW_OK;
}

static void gst_nvmm_tracker_set_property(GObject *o, guint id, const GValue *v,
                                          GParamSpec *p) {
    auto *self = GST_NVMM_TRACKER(o);
    if (id == PROP_IOU_THRESHOLD) self->iou_threshold = g_value_get_double(v);
    else if (id == PROP_MAX_AGE) self->max_age = g_value_get_int(v);
    else G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p);
}
static void gst_nvmm_tracker_get_property(GObject *o, guint id, GValue *v,
                                          GParamSpec *p) {
    auto *self = GST_NVMM_TRACKER(o);
    if (id == PROP_IOU_THRESHOLD) g_value_set_double(v, self->iou_threshold);
    else if (id == PROP_MAX_AGE) g_value_set_int(v, self->max_age);
    else G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p);
}

static void gst_nvmm_tracker_class_init(GstNvmmTrackerClass *klass) {
    auto *go = G_OBJECT_CLASS(klass);
    auto *el = GST_ELEMENT_CLASS(klass);
    auto *bt = GST_BASE_TRANSFORM_CLASS(klass);

    go->set_property = gst_nvmm_tracker_set_property;
    go->get_property = gst_nvmm_tracker_get_property;
    g_object_class_install_property(go, PROP_IOU_THRESHOLD,
        g_param_spec_double("iou-threshold", "IOU threshold",
            "Minimum IOU (same class) to continue a track", 0.0, 1.0, 0.3,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(go, PROP_MAX_AGE,
        g_param_spec_int("max-age", "Max age",
            "Frames a track survives with no match before it expires", 0, 100000, 30,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_add_static_pad_template(el, &sink_tmpl);
    gst_element_class_add_static_pad_template(el, &src_tmpl);
    gst_element_class_set_static_metadata(el,
        "NVMM Detection Tracker", "Filter/Analyzer/Video",
        "Assign a stable tracker_id to GstNvmmDetMeta detections by IOU matching "
        "across frames (no DeepStream, no CUDA)",
        "Pavel Guzenfeld");

    bt->start        = gst_nvmm_tracker_start;
    bt->stop         = gst_nvmm_tracker_stop;
    bt->transform_ip = gst_nvmm_tracker_transform_ip;
    bt->passthrough_on_same_caps = FALSE;  /* transform_ip must run every buffer */

    GST_DEBUG_CATEGORY_INIT(gst_nvmm_tracker_debug, "nvmmtracker", 0,
                            "NVMM detection tracker");
}

static void gst_nvmm_tracker_init(GstNvmmTracker *self) {
    self->iou_threshold = 0.3;
    self->max_age = 30;
    self->tracker = nullptr;
}

static gboolean plugin_init(GstPlugin *plugin) {
    return gst_element_register(plugin, "nvmmtracker", GST_RANK_NONE,
                                GST_TYPE_NVMM_TRACKER);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    nvmmtracker, "Assign tracker_id to NvmmDetMeta detections by IOU matching",
    plugin_init, PACKAGE_VERSION, "LGPL", "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp")
