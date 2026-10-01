#pragma once

#include <cstdint>

#include "shm_protocol.h"

namespace nvmm {

struct MotionEntry {
    float    mean_px;
    uint32_t moving;
};

uint32_t compute_box_motion(const int16_t *flow, int mv_w, int mv_h, int grid,
                            int frame_w, int frame_h,
                            const NvmmDetObject *objects, uint32_t n,
                            float threshold_px, MotionEntry *out);

}
