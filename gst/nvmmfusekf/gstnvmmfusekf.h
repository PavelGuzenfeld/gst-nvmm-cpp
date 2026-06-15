/// GstNvmmFuseKf — master constant-velocity Kalman filter that fuses the SAMURAI
/// track (GstNvmmTrackMeta) with the best YOLO detection (GstNvmmDetMeta) every
/// frame and writes the fused bbox back into the GstNvmmTrackMeta.
///
/// Pure-host, in-place passthrough on video/x-raw(memory:NVMM) — it only touches
/// metas (no CUDA/VIC), so it builds on every target. This is the "both detectors
/// complement each other" fusion: YOLO + SAMURAI run concurrently upstream; this
/// element is the top-level authority on the output box and (later) re-seeding.
#pragma once

#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_FUSEKF (gst_nvmm_fusekf_get_type())
G_DECLARE_FINAL_TYPE(GstNvmmFuseKf, gst_nvmm_fusekf, GST, NVMM_FUSEKF,
                     GstBaseTransform)

G_END_DECLS
