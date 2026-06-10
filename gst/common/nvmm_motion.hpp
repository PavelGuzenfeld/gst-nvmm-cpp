/// Per-detection motion from a dense optical-flow field (pure host, no CUDA).
///
/// The cross-modal Phase-3 payoff: given the OFA flow field (mv_w x mv_h cells,
/// each two int16 dx,dy in S10.5 fixed point — divide by 32 for pixels; one cell
/// covers grid x grid frame pixels) and the detector's boxes in frame pixels,
/// compute each box's mean flow magnitude and flag it moving when that exceeds
/// a threshold. Dependency-free so it is unit-tested on the host CI build.
#pragma once

#include <cstdint>

#include "shm_protocol.h"  // NvmmDetObject

namespace nvmm {

struct MotionEntry {
    float    mean_px;  // mean flow magnitude under the box, pixels/frame
    uint32_t moving;   // 1 when mean_px >= threshold
};

/// Fill `out[i]` for each of the `n` boxes. `flow` is the tightly-packed field
/// (mv_w*mv_h cells, 2x int16 each). Boxes are clamped to the frame; a box that
/// covers no whole cell falls back to its nearest cell. Returns n.
uint32_t compute_box_motion(const int16_t *flow, int mv_w, int mv_h, int grid,
                            int frame_w, int frame_h,
                            const NvmmDetObject *objects, uint32_t n,
                            float threshold_px, MotionEntry *out);

}  // namespace nvmm
