#include "config.h"

#include "nvmm_det_meta.h"
#include "nvmm_class_meta.h"

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_detlog_debug);
#define GST_CAT_DEFAULT gst_nvmm_detlog_debug

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

#define GST_TYPE_NVMM_DETLOG (gst_nvmm_detlog_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmDetLog, gst_nvmm_detlog, GST, NVMM_DETLOG, GstBaseTransform)

struct _GstNvmmDetLog {
    GstBaseTransform parent;
    gboolean per_object;
    guint64  frame_no;
};

G_DEFINE_TYPE(GstNvmmDetLog, gst_nvmm_detlog, GST_TYPE_BASE_TRANSFORM)

enum { PROP_0, PROP_PER_OBJECT };

static GstStaticPadTemplate sink_tmpl = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string)NV12, "
                    "width=(int)[32,8192], height=(int)[32,8192], "
                    "framerate=(fraction)[0/1, 240/1]"));
static GstStaticPadTemplate src_tmpl = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string)NV12, "
                    "width=(int)[32,8192], height=(int)[32,8192], "
                    "framerate=(fraction)[0/1, 240/1]"));

/// Returns GST_FLOW_OK with no metadata too: a passthrough must not stall.
static GstFlowReturn
gst_nvmm_detlog_transform_ip(GstBaseTransform *bt, GstBuffer *buf)
{
    auto *self = GST_NVMM_DETLOG(bt);
    const guint64 fno = self->frame_no++;

    GstNvmmDetMeta *det = gst_buffer_get_nvmm_det_meta(buf);
    if (!det) {
        GST_LOG_OBJECT(self, "frame %" G_GUINT64_FORMAT ": no det meta", fno);
        return GST_FLOW_OK;
    }
    GstNvmmClassMeta *cls = gst_buffer_get_nvmm_class_meta(buf);

    GST_INFO_OBJECT(self, "frame %" G_GUINT64_FORMAT ": %u object(s)%s",
                    fno, det->num_objects, cls ? " +class" : "");
    if (!self->per_object)
        return GST_FLOW_OK;

    for (guint32 i = 0; i < det->num_objects; i++) {
        const NvmmDetObject &o = det->objects[i];
        const NvmmClassEntry *c =
            (cls && i < cls->num_objects && cls->objects[i].class_id >= 0)
                ? &cls->objects[i] : nullptr;
        GST_INFO_OBJECT(self,
            "  [%u] %s #%" G_GUINT64_FORMAT " %.2f box=(%.0f,%.0f %.0fx%.0f)%s%s",
            i, o.label[0] ? o.label : "obj", o.tracker_id, o.confidence,
            o.left, o.top, o.width, o.height,
            c ? " class=" : "", c ? c->label : "");
    }
    return GST_FLOW_OK;
}

static void
gst_nvmm_detlog_set_property(GObject *o, guint id, const GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_DETLOG(o);
    if (id == PROP_PER_OBJECT) self->per_object = g_value_get_boolean(v);
    else G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p);
}

static void
gst_nvmm_detlog_get_property(GObject *o, guint id, GValue *v, GParamSpec *p)
{
    auto *self = GST_NVMM_DETLOG(o);
    if (id == PROP_PER_OBJECT) g_value_set_boolean(v, self->per_object);
    else G_OBJECT_WARN_INVALID_PROPERTY_ID(o, id, p);
}

static void
gst_nvmm_detlog_class_init(GstNvmmDetLogClass *klass)
{
    auto *go = G_OBJECT_CLASS(klass);
    auto *el = GST_ELEMENT_CLASS(klass);
    auto *bt = GST_BASE_TRANSFORM_CLASS(klass);

    go->set_property = gst_nvmm_detlog_set_property;
    go->get_property = gst_nvmm_detlog_get_property;
    g_object_class_install_property(go, PROP_PER_OBJECT,
        g_param_spec_boolean("per-object", "Per object",
            "Log every object, not just the per-frame summary", FALSE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_add_static_pad_template(el, &sink_tmpl);
    gst_element_class_add_static_pad_template(el, &src_tmpl);
    gst_element_class_set_static_metadata(el,
        "NVMM Detection Logger (example)", "Filter/Analyzer/Video",
        "Worked example element: passes NVMM frames through and logs the "
        "detection/classification metadata",
        "Pavel Guzenfeld");

    bt->transform_ip = gst_nvmm_detlog_transform_ip;
    bt->passthrough_on_same_caps = FALSE;

    GST_DEBUG_CATEGORY_INIT(gst_nvmm_detlog_debug, "nvmmdetlog", 0,
                            "NVMM detection logger example");
}

static void
gst_nvmm_detlog_init(GstNvmmDetLog *self)
{
    self->per_object = FALSE;
    self->frame_no = 0;
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
}

static gboolean
plugin_init(GstPlugin *plugin)
{
    return gst_element_register(plugin, "nvmmdetlog", GST_RANK_NONE,
                                GST_TYPE_NVMM_DETLOG);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    nvmmdetlog, "Worked example: log NVMM detection metadata",
    plugin_init, PACKAGE_VERSION, "LGPL", "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp")
