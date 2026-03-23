#include "gstnvmmappsrc.h"
#include "gstnvmmallocator.h"
#include "nvmm_types.hpp"

#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

#include "shm_protocol.h"

typedef NvmmShmHeader ShmHeader;

enum {
    PROP_0,
    PROP_SHM_NAME,
    PROP_IS_LIVE,
};

struct _GstNvmmAppSrcPrivate {
    std::string shm_name;
    int shm_fd;
    void *shm_ptr;
    gsize shm_size;
    uint64_t last_frame_number;
    GstVideoInfo video_info;
    gboolean is_live;
    gboolean caps_set;
};

G_DEFINE_TYPE_WITH_PRIVATE(GstNvmmAppSrc, gst_nvmm_app_src, GST_TYPE_PUSH_SRC)

static void
gst_nvmm_app_src_set_property(GObject *object, guint prop_id,
                               const GValue *value, GParamSpec *pspec)
{
    auto *self = GST_NVMM_APP_SRC(object);
    switch (prop_id) {
        case PROP_SHM_NAME:
            self->priv->shm_name = g_value_get_string(value) ? g_value_get_string(value) : "";
            break;
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

static gboolean
gst_nvmm_app_src_start(GstBaseSrc *src)
{
    auto *self = GST_NVMM_APP_SRC(src);

    if (self->priv->shm_name.empty()) {
        self->priv->shm_name = "/nvmm_sink_0";
    }

    self->priv->shm_fd = shm_open(self->priv->shm_name.c_str(), O_RDONLY, 0);
    if (self->priv->shm_fd < 0) {
        GST_ERROR_OBJECT(self, "shm_open(%s) failed: %s",
                         self->priv->shm_name.c_str(), strerror(errno));
        return FALSE;
    }

    struct stat st;
    if (fstat(self->priv->shm_fd, &st) < 0 || st.st_size == 0) {
        GST_ERROR_OBJECT(self, "Shared memory is empty or stat failed");
        close(self->priv->shm_fd);
        self->priv->shm_fd = -1;
        return FALSE;
    }
    self->priv->shm_size = st.st_size;

    self->priv->shm_ptr = mmap(NULL, self->priv->shm_size,
                                PROT_READ, MAP_SHARED,
                                self->priv->shm_fd, 0);
    if (self->priv->shm_ptr == MAP_FAILED) {
        GST_ERROR_OBJECT(self, "mmap failed: %s", strerror(errno));
        close(self->priv->shm_fd);
        self->priv->shm_fd = -1;
        self->priv->shm_ptr = nullptr;
        return FALSE;
    }

    self->priv->last_frame_number = 0;
    self->priv->caps_set = FALSE;

    GST_INFO_OBJECT(self, "Attached to shared memory '%s' (%" G_GSIZE_FORMAT " bytes)",
                    self->priv->shm_name.c_str(), self->priv->shm_size);
    return TRUE;
}

static gboolean
gst_nvmm_app_src_stop(GstBaseSrc *src)
{
    auto *self = GST_NVMM_APP_SRC(src);

    if (self->priv->shm_ptr && self->priv->shm_ptr != MAP_FAILED) {
        munmap(self->priv->shm_ptr, self->priv->shm_size);
        self->priv->shm_ptr = nullptr;
    }

    if (self->priv->shm_fd >= 0) {
        close(self->priv->shm_fd);
        self->priv->shm_fd = -1;
    }

    GST_INFO_OBJECT(self, "Detached from shared memory '%s'",
                    self->priv->shm_name.c_str());
    return TRUE;
}

static GstVideoFormat
shm_format_to_gst(uint32_t fmt)
{
    /* The sink writes the GstVideoFormat enum value directly,
       so we just cast it back. Validate it maps to a known string. */
    GstVideoFormat gst_fmt = static_cast<GstVideoFormat>(fmt);
    if (gst_video_format_to_string(gst_fmt) != nullptr) {
        return gst_fmt;
    }
    GST_WARNING("Unknown video format %u in shm header, defaulting to I420", fmt);
    return GST_VIDEO_FORMAT_I420;
}

static GstFlowReturn
gst_nvmm_app_src_create(GstPushSrc *push_src, GstBuffer **buf)
{
    auto *self = GST_NVMM_APP_SRC(push_src);
    auto *header = static_cast<const ShmHeader *>(self->priv->shm_ptr);

    if (!self->priv->shm_ptr) {
        return GST_FLOW_ERROR;
    }

    /* Wait for a new ready frame. Check flushing state to allow
       clean shutdown without blocking. */
    int attempts = 0;
    while (!header->ready || header->frame_number == self->priv->last_frame_number) {
        if (GST_PAD_IS_FLUSHING(GST_BASE_SRC_PAD(push_src))) {
            return GST_FLOW_FLUSHING;
        }
        if (++attempts > 500) {
            /* 500ms without a new frame — return EOS */
            GST_INFO_OBJECT(self, "No new frame for 500ms, returning EOS");
            return GST_FLOW_EOS;
        }
        g_usleep(1000);  /* 1ms */
    }

    __sync_synchronize();

    /* Validate header */
    if (header->magic != NVMM_SHM_MAGIC) {
        GST_ERROR_OBJECT(self, "Invalid shm magic: 0x%08x", header->magic);
        return GST_FLOW_ERROR;
    }

    /* Set caps on first frame if not already set */
    if (!self->priv->caps_set && header->width > 0 && header->height > 0) {
        GstVideoFormat fmt = shm_format_to_gst(header->format);
        gst_video_info_set_format(&self->priv->video_info, fmt,
                                  header->width, header->height);

        GstCaps *caps = gst_video_info_to_caps(&self->priv->video_info);
        gst_base_src_set_caps(GST_BASE_SRC(self), caps);
        gst_caps_unref(caps);
        self->priv->caps_set = TRUE;

        GST_INFO_OBJECT(self, "Caps from shm: %ux%u format=%s",
                        header->width, header->height,
                        gst_video_format_to_string(fmt));
    }

    /* Allocate output buffer and copy frame data */
    gsize data_size = header->data_size;
    if (data_size == 0 || data_size > self->priv->shm_size - sizeof(ShmHeader)) {
        GST_WARNING_OBJECT(self, "Invalid data_size: %u", header->data_size);
        return GST_FLOW_ERROR;
    }

    GstBuffer *buffer = gst_buffer_new_allocate(NULL, data_size, NULL);
    if (!buffer) {
        return GST_FLOW_ERROR;
    }

    GstMapInfo map_info;
    if (gst_buffer_map(buffer, &map_info, GST_MAP_WRITE)) {
        const auto *frame_data =
            static_cast<const uint8_t *>(self->priv->shm_ptr) + sizeof(ShmHeader);
        memcpy(map_info.data, frame_data, data_size);
        gst_buffer_unmap(buffer, &map_info);
    }

    GST_BUFFER_PTS(buffer) = header->timestamp_ns;
    GST_BUFFER_DURATION(buffer) = GST_CLOCK_TIME_NONE;

    self->priv->last_frame_number = header->frame_number;

    *buf = buffer;
    return GST_FLOW_OK;
}

static void
gst_nvmm_app_src_finalize(GObject *object)
{
    auto *self = GST_NVMM_APP_SRC(object);
    self->priv->~_GstNvmmAppSrcPrivate();
    G_OBJECT_CLASS(gst_nvmm_app_src_parent_class)->finalize(object);
}

static void
gst_nvmm_app_src_class_init(GstNvmmAppSrcClass *klass)
{
    auto *gobject_class = G_OBJECT_CLASS(klass);
    auto *element_class = GST_ELEMENT_CLASS(klass);
    auto *basesrc_class = GST_BASE_SRC_CLASS(klass);
    auto *pushsrc_class = GST_PUSH_SRC_CLASS(klass);

    gobject_class->set_property = gst_nvmm_app_src_set_property;
    gobject_class->get_property = gst_nvmm_app_src_get_property;
    gobject_class->finalize = gst_nvmm_app_src_finalize;

    g_object_class_install_property(gobject_class, PROP_SHM_NAME,
        g_param_spec_string("shm-name", "Shared Memory Name",
            "POSIX shared memory segment name to read from",
            "/nvmm_sink_0", G_PARAM_READWRITE));

    g_object_class_install_property(gobject_class, PROP_IS_LIVE,
        g_param_spec_boolean("is-live", "Is Live",
            "Whether this source is a live source",
            TRUE, G_PARAM_READWRITE));

    gst_element_class_set_static_metadata(element_class,
        "NVMM Shared Memory Source",
        "Source/Video",
        "Reads NVMM video frames from POSIX shared memory for zero-copy IPC",
        "Pavel Guzenfeld");

    GstCaps *caps = gst_caps_from_string(
        "video/x-raw, "
        "format=(string){NV12, RGBA, I420, BGRA}, "
        "width=(int)[1, 8192], height=(int)[1, 8192], "
        "framerate=(fraction)[0/1, 240/1]");
    gst_element_class_add_pad_template(element_class,
        gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
    gst_caps_unref(caps);

    basesrc_class->start = gst_nvmm_app_src_start;
    basesrc_class->stop = gst_nvmm_app_src_stop;
    pushsrc_class->create = gst_nvmm_app_src_create;
}

static void
gst_nvmm_app_src_init(GstNvmmAppSrc *self)
{
    self->priv = static_cast<GstNvmmAppSrcPrivate *>(
        gst_nvmm_app_src_get_instance_private(self));
    new (self->priv) GstNvmmAppSrcPrivate();
    self->priv->shm_name = "/nvmm_sink_0";
    self->priv->shm_fd = -1;
    self->priv->shm_ptr = nullptr;
    self->priv->shm_size = 0;
    self->priv->last_frame_number = 0;
    self->priv->is_live = TRUE;
    self->priv->caps_set = FALSE;

    gst_base_src_set_live(GST_BASE_SRC(self), TRUE);
    gst_base_src_set_format(GST_BASE_SRC(self), GST_FORMAT_TIME);
}

/* Plugin registration */
static gboolean
plugin_init(GstPlugin *plugin)
{
    return gst_element_register(plugin, "nvmmappsrc",
                                GST_RANK_NONE, GST_TYPE_NVMM_APP_SRC);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvmmappsrc,
    "NVMM shared memory source for zero-copy IPC",
    plugin_init,
    "1.0.1",
    "LGPL",
    "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp"
)
