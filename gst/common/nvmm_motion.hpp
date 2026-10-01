#pragma once

#include <cstdint>

#include "shm_protocol.h"

namespace nvmm {

struct MotionEntry {
    /// Pixels per frame.
    float    mean_px;
    uint32_t moving;
};

/// `flow` is mv_w*mv_h cells of int16 (dx, dy) in S10.5 fixed point, one cell
/// per grid x grid frame pixels. Boxes are in frame pixels; one covering no
/// whole cell uses its nearest cell.
uint32_t compute_box_motion(const int16_t *flow, int mv_w, int mv_h, int grid,
                            int frame_w, int frame_h,
                            const NvmmDetObject *objects, uint32_t n,
                            float threshold_px, MotionEntry *out);

}
