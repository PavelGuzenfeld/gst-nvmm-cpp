/// IPC backend interface — abstract producer/consumer operations.
///
/// Two implementations live alongside each other; meson picks one at compile
/// time via has_header_symbol(nvbufsurface.h, NvBufSurfaceMapParams):
///
///   * gst/ipc_jp5/ipc_jp5.cpp
///       Shared-memory + CPU copy. Wire format: NvmmShmCopyHeader.
///       Targets JetPack 5 / L4T 35.x where NvBufSurfaceImport is missing.
///
///   * gst/ipc_jp6/ipc_jp6.cpp
///       Pool + SCM_RIGHTS fd passing; consumer-side zero-copy via
///       NvBufSurfaceImport. Wire format: NvmmShmPoolHeader. Optionally
///       offers its NVMM pool to upstream via propose_allocation for true
///       end-to-end zero-copy. Targets JetPack 6 / L4T 36.x+.
///
/// The GStreamer element files (nvmmsink, nvmmappsrc) call only this
/// interface — they never #ifdef on backend choice.
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
/// `pool_size` is advisory — the JP5 (copy) backend ignores it; the JP6 (pool) backend uses it for the NVMM pool
/// (clamped to backend limits).
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

/// Optional pool proposal for upstream — the JP6 (pool) backend implements true zero-copy here,
/// the JP5 (copy) backend no-ops and returns FALSE.
gboolean         nvmm_ipc_producer_propose_allocation(NvmmIpcProducer *self,
                                                      GstElement       *owner,
                                                      GstQuery         *query);

/// Render a frame. The buffer is borrowed for the duration of the call in the JP5 backend;
/// the JP6 backend may retain it asynchronously until remote consumers release their
/// references. Either way the caller hands ownership of the ref to the backend
/// via gst_buffer_ref — the backend unrefs when done.
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
/// returned ref and must gst_buffer_unref when done.
///
/// On the JP5 backend the buffer contains a CPU copy of the frame.
/// On the JP6 backend the buffer wraps an imported NVMM surface (zero-copy).
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
