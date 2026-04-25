/// IPC backend interface — abstract producer/consumer operations.
///
/// Single implementation: gst/ipc_pool/ipc_pool.cpp. Pool + SCM_RIGHTS fd
/// passing, consumer-side zero-copy via NvBufSurfaceImport. Wire format:
/// NvmmShmPoolHeader.
///
/// Requires the NVMM cross-process import API (NvBufSurfaceImport,
/// NvBufSurfaceGetMapParams, NvBufSurfaceMapParams). Available on:
///   * L4T R35.3.1 (JetPack 5.1.1, March 2023) and later for the JP5 line
///   * L4T R36.0   (JetPack 6.0)               and later for the JP6 line
/// meson rejects older toolchains at configure time.
///
/// The GStreamer element files (nvmmsink, nvmmappsrc) call only this
/// interface — they have no JetPack-version conditionals.
#pragma once

#include <gst/gst.h>
#include <gst/base/base.h>
#include <gst/video/video.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle types — each backend defines the concrete struct. */
typedef struct NvmmIpcProducer NvmmIpcProducer;
typedef struct NvmmIpcConsumer NvmmIpcConsumer;

/// Register the backend's own GST_DEBUG_CATEGORY. Plugins must call this
/// from their plugin_init — once per plugin is enough.
void nvmm_ipc_backend_init_debug(void);

/* ------------------------------------------------------------------ */
/*                            Producer API                            */
/* ------------------------------------------------------------------ */

/// Create a producer bound to a POSIX shared-memory segment name.
/// `pool_size` is the number of NVMM buffers in the cross-process pool,
/// clamped to [4, NVMM_POOL_SIZE_MAX].
NvmmIpcProducer *nvmm_ipc_producer_new(const gchar *shm_name, int pool_size);
void             nvmm_ipc_producer_free(NvmmIpcProducer *self);

/// Allocate backing resources. Must be called after construction and before
/// set_caps / render.
gboolean         nvmm_ipc_producer_start(NvmmIpcProducer *self, GstElement *owner);
gboolean         nvmm_ipc_producer_stop (NvmmIpcProducer *self, GstElement *owner);

/// Called from the sink's set_caps vfunc.
gboolean         nvmm_ipc_producer_set_caps(NvmmIpcProducer *self,
                                            GstElement       *owner,
                                            const GstVideoInfo *info,
                                            gboolean            caps_have_nvmm_feature);

/// Pool proposal hook for upstream — when implemented, lets upstream
/// allocate directly into our pool slots so render() can skip the GPU-to-GPU
/// NvBufSurfaceCopy. Currently returns FALSE; render always copies.
gboolean         nvmm_ipc_producer_propose_allocation(NvmmIpcProducer *self,
                                                      GstElement       *owner,
                                                      GstQuery         *query);

/// Render a frame. The backend may retain the buffer asynchronously until
/// remote consumers release their references; the caller hands the ref
/// over and the backend unrefs when done.
GstFlowReturn    nvmm_ipc_producer_render(NvmmIpcProducer *self,
                                          GstElement       *owner,
                                          GstBuffer        *buffer);

/* ------------------------------------------------------------------ */
/*                            Consumer API                            */
/* ------------------------------------------------------------------ */

NvmmIpcConsumer *nvmm_ipc_consumer_new (const gchar *shm_name);
void             nvmm_ipc_consumer_free(NvmmIpcConsumer *self);

gboolean         nvmm_ipc_consumer_start(NvmmIpcConsumer *self, GstElement *owner);
gboolean         nvmm_ipc_consumer_stop (NvmmIpcConsumer *self, GstElement *owner);

/// Block (with 1ms polling) until a new frame is available or the pad becomes
/// flushing, then return a GstBuffer representing the frame. Caller owns the
/// returned ref and must gst_buffer_unref when done. The buffer wraps an
/// imported NVMM surface — consumer-side reads are zero-copy from GPU memory.
GstFlowReturn    nvmm_ipc_consumer_fetch(NvmmIpcConsumer *self,
                                         GstElement       *owner,
                                         GstPad           *src_pad,
                                         GstBuffer       **out_buffer);

/// Fetch the currently-advertised GstVideoInfo from the producer header.
/// Returns FALSE until the producer has published its first frame.
gboolean         nvmm_ipc_consumer_peek_caps(NvmmIpcConsumer *self,
                                             GstVideoInfo    *out_info,
                                             gboolean        *out_is_nvmm);

#ifdef __cplusplus
}
#endif
