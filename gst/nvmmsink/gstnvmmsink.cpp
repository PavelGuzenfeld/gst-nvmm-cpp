/// GstNvmmSink — thin GStreamer element that delegates all IPC to the
/// pool + SCM_RIGHTS backend (ipc_pool).
///
/// Element responsibilities:
///   - GObject property plumbing (shm-name, pool-size)
///   - caps negotiation and set_caps parsing
///   - propose_allocation hand-off to the backend (backend may offer its pool)
///   - render() forwards to the backend
/// IPC implementation lives in gst/ipc_pool/ipc_pool.cpp.

#include "gstnvmmsink.h"
#include "ipc_backend.h"
#include "nvmm_config.h"
#include "shm_protocol.h"   /* NVMM_POOL_SIZE_MAX */

#include <string>

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_ipc_debug);
#define GST_CAT_DEFAULT gst_nvmm_ipc_debug

enum {
    PROP_0,
    PROP_SHM_NAME,
    PROP_POOL_SIZE,
};


struct _GstNvmmSinkPrivate {
    std::string       shm_name;
    int               pool_size;
    NvmmIpcProducer  *prod;
};

G_DEFINE_TYPE_WITH_PRIVATE(GstNvmmSink, gst_nvmm_sink, GST_TYPE_BASE_SINK)

/* --- GObject properties ------------------------------------------ */

static void
gst_nvmm_sink_set_property(GObject *object, guint prop_id,
                           const GValue *value, GParamSpec *pspec)
{
    auto *self = GST_NVMM_SINK(object);
    switch (prop_id) {
        case PROP_SHM_NAME: {
            const gchar *s = g_value_get_string(value);
            self->priv->shm_name = s ? s : "";
            break;
        }
        case PROP_POOL_SIZE:
            self->priv->pool_size =
                CLAMP(g_value_get_int(value), nvmm::config::min_pool_size, NVMM_POOL_SIZE_MAX);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void
gst_nvmm_sink_get_property(GObject *object, guint prop_id,
                           GValue *value, GParamSpec *pspec)
{
    auto *self = GST_NVMM_SINK(object);
    switch (prop_id) {
        case PROP_SHM_NAME:
            g_value_set_string(value, self->priv->shm_name.c_str());
            break;
        case PROP_POOL_SIZE:
            g_value_set_int(value, self->priv->pool_size);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

/* --- GstBaseSink vfuncs ------------------------------------------ */

static gboolean
gst_nvmm_sink_start(GstBaseSink *sink)
{
    auto *self = GST_NVMM_SINK(sink);
    if (self->priv->shm_name.empty())
        self->priv->shm_name = nvmm::config::default_shm_name;

    self->priv->prod = nvmm_ipc_producer_new(self->priv->shm_name.c_str(),
                                              self->priv->pool_size);
    if (!self->priv->prod) return FALSE;
    return nvmm_ipc_producer_start(self->priv->prod, GST_ELEMENT(self));
}

static gboolean
gst_nvmm_sink_stop(GstBaseSink *sink)
{
    auto *self = GST_NVMM_SINK(sink);
    gboolean ok = TRUE;
    if (self->priv->prod) {
        ok = nvmm_ipc_producer_stop(self->priv->prod, GST_ELEMENT(self));
        nvmm_ipc_producer_free(self->priv->prod);
        self->priv->prod = nullptr;
    }
    return ok;
}

static gboolean
gst_nvmm_sink_set_caps(GstBaseSink *sink, GstCaps *caps)
{
    auto *self = GST_NVMM_SINK(sink);
    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, caps)) {
        GST_ERROR_OBJECT(self, "failed to parse caps");
        return FALSE;
    }

    GstCapsFeatures *feat = gst_caps_get_features(caps, 0);
    gboolean is_nvmm = feat &&
        gst_caps_features_contains(feat, "memory:NVMM");

    return nvmm_ipc_producer_set_caps(self->priv->prod, GST_ELEMENT(self),
                                      &info, is_nvmm);
}

static gboolean
gst_nvmm_sink_propose_allocation(GstBaseSink *sink, GstQuery *query)
{
    auto *self = GST_NVMM_SINK(sink);
    if (nvmm_ipc_producer_propose_allocation(self->priv->prod,
                                             GST_ELEMENT(self), query))
        return TRUE;

    /* Chain to parent if it has one — GstBaseSink's default propose_allocation
     * vtable slot may be NULL, in which case just return TRUE (nothing to
     * propose). */
    auto *parent_class = GST_BASE_SINK_CLASS(gst_nvmm_sink_parent_class);
    if (parent_class && parent_class->propose_allocation)
        return parent_class->propose_allocation(sink, query);
    return TRUE;
}

static GstFlowReturn
gst_nvmm_sink_render(GstBaseSink *sink, GstBuffer *buffer)
{
    auto *self = GST_NVMM_SINK(sink);
    return nvmm_ipc_producer_render(self->priv->prod, GST_ELEMENT(self), buffer);
}

/* --- Class / instance init --------------------------------------- */

static void
gst_nvmm_sink_class_init(GstNvmmSinkClass *klass)
{
    auto *gobject_class = G_OBJECT_CLASS(klass);
    auto *element_class = GST_ELEMENT_CLASS(klass);
    auto *basesink_class = GST_BASE_SINK_CLASS(klass);

    gobject_class->set_property = gst_nvmm_sink_set_property;
    gobject_class->get_property = gst_nvmm_sink_get_property;

    basesink_class->start              = gst_nvmm_sink_start;
    basesink_class->stop               = gst_nvmm_sink_stop;
    basesink_class->set_caps           = gst_nvmm_sink_set_caps;
    basesink_class->propose_allocation = gst_nvmm_sink_propose_allocation;
    basesink_class->render             = gst_nvmm_sink_render;

    g_object_class_install_property(gobject_class, PROP_SHM_NAME,
        g_param_spec_string("shm-name", "Shared Memory Name",
            "POSIX shared-memory segment name (e.g. /cam1)",
            nvmm::config::default_shm_name, G_PARAM_READWRITE));

    g_object_class_install_property(gobject_class, PROP_POOL_SIZE,
        g_param_spec_int("pool-size", "Pool Size",
            "Number of NVMM buffers in the cross-process pool",
            nvmm::config::min_pool_size, NVMM_POOL_SIZE_MAX, nvmm::config::default_pool_size,
            G_PARAM_READWRITE));

    gst_element_class_set_static_metadata(element_class,
        "NVMM IPC Sink",
        "Sink/Video",
        "Shares NVMM video frames with out-of-process consumers via "
        "an NVMM buffer pool and SCM_RIGHTS fd passing. Consumer-side "
        "zero-copy (NvBufSurfaceImport). Requires JetPack 5.1.1+ or 6.0+.",
        "Pavel Guzenfeld, Stereolabs");

    GstCaps *caps = gst_caps_from_string(
        "video/x-raw(memory:NVMM), "
        "format=(string){NV12, RGBA, I420, BGRA}, "
        "width=(int)[1, 8192], height=(int)[1, 8192], "
        "framerate=(fraction)[0/1, 240/1]");
    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps));
    gst_caps_unref(caps);
}

static void
gst_nvmm_sink_init(GstNvmmSink *self)
{
    self->priv = static_cast<GstNvmmSinkPrivate *>(
        gst_nvmm_sink_get_instance_private(self));
    new (self->priv) GstNvmmSinkPrivate();
    self->priv->shm_name = nvmm::config::default_shm_name;
    self->priv->pool_size = nvmm::config::default_pool_size;
    self->priv->prod = nullptr;
}

/* --- Plugin registration ----------------------------------------- */

static gboolean
plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_nvmm_ipc_debug, "nvmmsink", 0, "NVMM IPC sink");
    nvmm_ipc_backend_init_debug();
    return gst_element_register(plugin, "nvmmsink", GST_RANK_NONE,
                                GST_TYPE_NVMM_SINK);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    nvmmsink,
    "NVMM IPC sink",
    plugin_init,
    "1.2.0",
    "LGPL",
    "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp"
)
