#include "gstnvmmbufferpool.h"
#include "gstnvmmallocator.h"
#include "nvmm_buffer.hpp"
#include "nvmm_types.hpp"

struct _GstNvmmBufferPoolPrivate {
    GstVideoInfo video_info;
    GstAllocator* allocator;
    gboolean configured;
};

G_DEFINE_TYPE_WITH_PRIVATE(GstNvmmBufferPool, gst_nvmm_buffer_pool,
                           GST_TYPE_BUFFER_POOL)

static const gchar**
gst_nvmm_buffer_pool_get_options(GstBufferPool* pool)
{
    (void)pool;
    static const gchar* options[] = {
        GST_BUFFER_POOL_OPTION_VIDEO_META,
        NULL
    };
    return options;
}

static nvmm::ColorFormat
gst_format_to_nvmm(GstVideoFormat fmt)
{
    switch (fmt) {
        case GST_VIDEO_FORMAT_NV12: return nvmm::ColorFormat::kNV12;
        case GST_VIDEO_FORMAT_RGBA: return nvmm::ColorFormat::kRGBA;
        case GST_VIDEO_FORMAT_BGRA: return nvmm::ColorFormat::kBGRA;
        case GST_VIDEO_FORMAT_I420: return nvmm::ColorFormat::kI420;
        default:                    return nvmm::ColorFormat::kNV12;
    }
}

static gboolean
gst_nvmm_buffer_pool_set_config(GstBufferPool* pool, GstStructure* config)
{
    auto* self = GST_NVMM_BUFFER_POOL(pool);
    GstCaps* caps = NULL;
    guint size, min_buffers, max_buffers;

    if (!gst_buffer_pool_config_get_params(config, &caps, &size,
                                            &min_buffers, &max_buffers)) {
        GST_ERROR_OBJECT(pool, "Failed to get pool config params");
        return FALSE;
    }

    if (!caps) {
        GST_ERROR_OBJECT(pool, "No caps in pool config");
        return FALSE;
    }

    if (!gst_video_info_from_caps(&self->priv->video_info, caps)) {
        GST_ERROR_OBJECT(pool, "Failed to parse caps");
        return FALSE;
    }

    /* Create our NVMM allocator */
    if (self->priv->allocator) {
        gst_object_unref(self->priv->allocator);
    }
    self->priv->allocator = gst_nvmm_allocator_new(0 /* default */);

    /* Update config with actual NVMM buffer size */
    gsize nvmm_size = GST_VIDEO_INFO_SIZE(&self->priv->video_info);
    gst_buffer_pool_config_set_params(config, caps, nvmm_size,
                                       min_buffers, max_buffers);

    self->priv->configured = TRUE;

    GST_INFO_OBJECT(pool, "Configured: %dx%d %s, %u-%u buffers",
                    GST_VIDEO_INFO_WIDTH(&self->priv->video_info),
                    GST_VIDEO_INFO_HEIGHT(&self->priv->video_info),
                    gst_video_format_to_string(
                        GST_VIDEO_INFO_FORMAT(&self->priv->video_info)),
                    min_buffers, max_buffers);

    return GST_BUFFER_POOL_CLASS(gst_nvmm_buffer_pool_parent_class)
        ->set_config(pool, config);
}

static GstFlowReturn
gst_nvmm_buffer_pool_alloc(GstBufferPool* pool, GstBuffer** buffer,
                            GstBufferPoolAcquireParams* params)
{
    (void)params;
    auto* self = GST_NVMM_BUFFER_POOL(pool);

    if (!self->priv->configured) {
        GST_ERROR_OBJECT(pool, "Pool not configured");
        return GST_FLOW_ERROR;
    }

    gint w = GST_VIDEO_INFO_WIDTH(&self->priv->video_info);
    gint h = GST_VIDEO_INFO_HEIGHT(&self->priv->video_info);
    GstVideoFormat fmt = GST_VIDEO_INFO_FORMAT(&self->priv->video_info);

    /* Create NvBufSurface with exact format/dimensions */
    nvmm::SurfaceParams sp;
    sp.width = static_cast<uint32_t>(w);
    sp.height = static_cast<uint32_t>(h);
    sp.color_format = gst_format_to_nvmm(fmt);
    sp.mem_type = nvmm::MemoryType::kDefault;

    auto result = nvmm::NvmmBuffer::create(sp);
    if (!result) {
        GST_ERROR_OBJECT(pool, "NvBufSurfaceCreate failed");
        return GST_FLOW_ERROR;
    }

    gsize buf_size = static_cast<gsize>(result.value().data_size());

    /* Allocate GstMemory via our allocator (wraps the surface) */
    GstMemory* mem = gst_allocator_alloc(self->priv->allocator, buf_size, NULL);
    if (!mem) {
        GST_ERROR_OBJECT(pool, "Failed to allocate NVMM GstMemory");
        return GST_FLOW_ERROR;
    }

    /* Build the GstBuffer */
    *buffer = gst_buffer_new();
    gst_buffer_append_memory(*buffer, mem);

    /* Add video meta with correct plane info */
    gst_buffer_add_video_meta_full(*buffer, GST_VIDEO_FRAME_FLAG_NONE,
        fmt, w, h,
        GST_VIDEO_INFO_N_PLANES(&self->priv->video_info),
        self->priv->video_info.offset,
        self->priv->video_info.stride);

    return GST_FLOW_OK;
}

static void
gst_nvmm_buffer_pool_dispose(GObject* object)
{
    auto* self = GST_NVMM_BUFFER_POOL(object);
    if (self->priv->allocator) {
        gst_object_unref(self->priv->allocator);
        self->priv->allocator = NULL;
    }
    G_OBJECT_CLASS(gst_nvmm_buffer_pool_parent_class)->dispose(object);
}

static void
gst_nvmm_buffer_pool_class_init(GstNvmmBufferPoolClass* klass)
{
    auto* gobject_class = G_OBJECT_CLASS(klass);
    auto* pool_class = GST_BUFFER_POOL_CLASS(klass);

    gobject_class->dispose = gst_nvmm_buffer_pool_dispose;

    pool_class->get_options = gst_nvmm_buffer_pool_get_options;
    pool_class->set_config = gst_nvmm_buffer_pool_set_config;
    pool_class->alloc_buffer = gst_nvmm_buffer_pool_alloc;
}

static void
gst_nvmm_buffer_pool_init(GstNvmmBufferPool* self)
{
    self->priv = static_cast<GstNvmmBufferPoolPrivate*>(
        gst_nvmm_buffer_pool_get_instance_private(self));
    self->priv->allocator = NULL;
    self->priv->configured = FALSE;
}

/* Public API */

GstBufferPool*
gst_nvmm_buffer_pool_new(void)
{
    return GST_BUFFER_POOL(g_object_new(GST_TYPE_NVMM_BUFFER_POOL, NULL));
}
