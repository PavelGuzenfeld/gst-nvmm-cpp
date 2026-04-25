/// GstNvmmAppSrc — thin GStreamer element that delegates all IPC to the
/// pool + SCM_RIGHTS backend (gst/ipc_pool/ipc_pool.cpp). The element
/// handles property plumbing, caps propagation, and buffer forwarding;
/// the backend owns the wire format and transport.

#include "gstnvmmappsrc.h"
#include "ipc_backend.h"
#include "nvmm_config.h"

#include <string>

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

GST_DEBUG_CATEGORY_STATIC(gst_nvmm_appsrc_debug);
#define GST_CAT_DEFAULT gst_nvmm_appsrc_debug

enum {
    PROP_0,
    PROP_SHM_NAME,
    PROP_IS_LIVE,
};

struct _GstNvmmAppSrcPrivate {
    std::string       shm_name;
    gboolean          is_live;
    NvmmIpcConsumer  *cons;
};

G_DEFINE_TYPE_WITH_PRIVATE(GstNvmmAppSrc, gst_nvmm_app_src, GST_TYPE_PUSH_SRC)

/* --- GObject properties ------------------------------------------ */

static void
gst_nvmm_app_src_set_property(GObject *object, guint prop_id,
                              const GValue *value, GParamSpec *pspec)
{
    auto *self = GST_NVMM_APP_SRC(object);
    switch (prop_id) {
        case PROP_SHM_NAME: {
            const gchar *s = g_value_get_string(value);
            self->priv->shm_name = s ? s : "";
            break;
        }
        case PROP_IS_LIVE:
            self->priv->is_live = g_value_get_boolean(value);
            gst_base_src_set_live(GST_BASE_SRC(self), self->priv->is_live);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void
gst_nvmm_app_src_get_property(GObject *object, guint prop_id,
                              GValue *value, GParamSpec *pspec)
{
    auto *self = GST_NVMM_APP_SRC(object);
    switch (prop_id) {
        case PROP_SHM_NAME:
            g_value_set_string(value, self->priv->shm_name.c_str());
            break;
        case PROP_IS_LIVE:
            g_value_set_boolean(value, self->priv->is_live);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

/* --- GstBaseSrc vfuncs ------------------------------------------- */

static gboolean
gst_nvmm_app_src_start(GstBaseSrc *src)
{
    auto *self = GST_NVMM_APP_SRC(src);
    if (self->priv->shm_name.empty())
        self->priv->shm_name = nvmm::config::default_shm_name;

    self->priv->cons = nvmm_ipc_consumer_new(self->priv->shm_name.c_str());
    return nvmm_ipc_consumer_start(self->priv->cons, GST_ELEMENT(self));
}

static gboolean
gst_nvmm_app_src_stop(GstBaseSrc *src)
{
    auto *self = GST_NVMM_APP_SRC(src);
    gboolean ok = TRUE;
    if (self->priv->cons) {
        ok = nvmm_ipc_consumer_stop(self->priv->cons, GST_ELEMENT(self));
        nvmm_ipc_consumer_free(self->priv->cons);
        self->priv->cons = nullptr;
    }
    return ok;
}

static GstFlowReturn
gst_nvmm_app_src_create(GstPushSrc *push_src, GstBuffer **buf)
{
    auto *self = GST_NVMM_APP_SRC(push_src);
    return nvmm_ipc_consumer_fetch(self->priv->cons, GST_ELEMENT(self),
                                   GST_BASE_SRC_PAD(push_src), buf);
}

/* --- Class / instance init --------------------------------------- */

static void
gst_nvmm_app_src_class_init(GstNvmmAppSrcClass *klass)
{
    auto *gobject_class = G_OBJECT_CLASS(klass);
    auto *element_class = GST_ELEMENT_CLASS(klass);
    auto *basesrc_class = GST_BASE_SRC_CLASS(klass);
    auto *pushsrc_class = GST_PUSH_SRC_CLASS(klass);

    gobject_class->set_property = gst_nvmm_app_src_set_property;
    gobject_class->get_property = gst_nvmm_app_src_get_property;

    basesrc_class->start = gst_nvmm_app_src_start;
    basesrc_class->stop  = gst_nvmm_app_src_stop;
    pushsrc_class->create = gst_nvmm_app_src_create;

    g_object_class_install_property(gobject_class, PROP_SHM_NAME,
        g_param_spec_string("shm-name", "Shared Memory Name",
            "POSIX shared-memory segment name (must match the sink)",
            nvmm::config::default_shm_name, G_PARAM_READWRITE));

    g_object_class_install_property(gobject_class, PROP_IS_LIVE,
        g_param_spec_boolean("is-live", "Is Live",
            "Advertise this source as live",
            TRUE, G_PARAM_READWRITE));

    gst_element_class_set_static_metadata(element_class,
        "NVMM IPC Source",
        "Source/Video",
        "Reads NVMM video frames shared by nvmmsink via an NVMM buffer pool "
        "and SCM_RIGHTS fd passing. Consumer-side zero-copy "
        "(NvBufSurfaceImport). Requires JetPack 5.1.1+ or 6.0+.",
        "Pavel Guzenfeld, Stereolabs");

    GstCaps *caps = gst_caps_from_string(
        "video/x-raw(memory:NVMM), "
        "format=(string){NV12, RGBA, I420, BGRA}, "
        "width=(int)[1, 8192], height=(int)[1, 8192], "
        "framerate=(fraction)[0/1, 240/1]; "
        "video/x-raw, "
        "format=(string){NV12, RGBA, I420, BGRA}, "
        "width=(int)[1, 8192], height=(int)[1, 8192], "
        "framerate=(fraction)[0/1, 240/1]");
    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
    gst_caps_unref(caps);
}

static void
gst_nvmm_app_src_init(GstNvmmAppSrc *self)
{
    self->priv = static_cast<GstNvmmAppSrcPrivate *>(
        gst_nvmm_app_src_get_instance_private(self));
    new (self->priv) GstNvmmAppSrcPrivate();
    self->priv->shm_name = nvmm::config::default_shm_name;
    self->priv->is_live  = TRUE;
    self->priv->cons     = nullptr;

    gst_base_src_set_format(GST_BASE_SRC(self), GST_FORMAT_TIME);
    gst_base_src_set_live(GST_BASE_SRC(self), TRUE);
}

/* --- Plugin registration ----------------------------------------- */

static gboolean
plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_nvmm_appsrc_debug, "nvmmappsrc", 0,
                            "NVMM IPC source");
    nvmm_ipc_backend_init_debug();
    return gst_element_register(plugin, "nvmmappsrc", GST_RANK_NONE,
                                GST_TYPE_NVMM_APP_SRC);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR, GST_VERSION_MINOR,
    nvmmappsrc,
    "NVMM IPC source",
    plugin_init,
    "1.2.0",
    "LGPL",
    "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp"
)
