#include "nvmm_motion.hpp"

#include <algorithm>
#include <cmath>

namespace nvmm {

uint32_t compute_box_motion(const int16_t *flow, int mv_w, int mv_h, int grid,
                            int frame_w, int frame_h,
                            const NvmmDetObject *objects, uint32_t n,
                            float threshold_px, MotionEntry *out) {
    if (!flow || mv_w <= 0 || mv_h <= 0 || grid <= 0 || !objects || !out || n == 0)
        return 0;
    constexpr float kS105 = 1.0f / 32.0f;  // S10.5 fixed point -> pixels

    for (uint32_t i = 0; i < n; i++) {
        const NvmmDetObject &o = objects[i];
        // Frame-pixel box -> inclusive cell range, clamped to the field. A box
        // smaller than a cell (or clamped to an edge) still covers >= 1 cell.
        const float x1 = std::min(std::max(o.left, 0.f), (float)frame_w - 1);
        const float y1 = std::min(std::max(o.top, 0.f), (float)frame_h - 1);
        const float x2 = std::min(std::max(o.left + o.width - 1, x1), (float)frame_w - 1);
        const float y2 = std::min(std::max(o.top + o.height - 1, y1), (float)frame_h - 1);
        const int cx1 = std::min((int)(x1 / grid), mv_w - 1);
        const int cy1 = std::min((int)(y1 / grid), mv_h - 1);
        const int cx2 = std::min((int)(x2 / grid), mv_w - 1);
        const int cy2 = std::min((int)(y2 / grid), mv_h - 1);

        double sum = 0.0;
        for (int cy = cy1; cy <= cy2; cy++) {
            const int16_t *row = flow + ((size_t)cy * mv_w + cx1) * 2;
            for (int cx = cx1; cx <= cx2; cx++, row += 2) {
                const float dx = row[0] * kS105, dy = row[1] * kS105;
                sum += std::sqrt((double)dx * dx + (double)dy * dy);
            }
        }
        const int cells = (cx2 - cx1 + 1) * (cy2 - cy1 + 1);
        const float mean = (float)(sum / cells);
        out[i].mean_px = mean;
        out[i].moving = mean >= threshold_px ? 1u : 0u;
    }
    return n;
}

}  // namespace nvmm
