/// GstNvmmBufferPool — GStreamer buffer pool for NVMM memory.
/// Pre-allocates NvBufSurface buffers and recycles them to avoid
/// per-frame allocation overhead.
#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_BUFFER_POOL (gst_nvmm_buffer_pool_get_type())
#define GST_NVMM_BUFFER_POOL(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVMM_BUFFER_POOL, GstNvmmBufferPool))
#define GST_IS_NVMM_BUFFER_POOL(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVMM_BUFFER_POOL))

typedef struct _GstNvmmBufferPool GstNvmmBufferPool;
typedef struct _GstNvmmBufferPoolClass GstNvmmBufferPoolClass;
typedef struct _GstNvmmBufferPoolPrivate GstNvmmBufferPoolPrivate;

struct _GstNvmmBufferPool {
    GstBufferPool parent;
    GstNvmmBufferPoolPrivate* priv;
};

struct _GstNvmmBufferPoolClass {
    GstBufferPoolClass parent_class;
};

GType gst_nvmm_buffer_pool_get_type(void);

/// Create a new NVMM buffer pool.
GstBufferPool* gst_nvmm_buffer_pool_new(void);

G_END_DECLS
