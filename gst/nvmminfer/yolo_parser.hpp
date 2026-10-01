#pragma once

#include <cstdint>
#include <vector>

#include "shm_protocol.h"

namespace nvmm {

/// frame_x = (net_x - pad_x) / scale; frame_w/h is the det-meta coordinate space.
struct LetterboxInfo {
    float scale = 1.f;
    float pad_x = 0.f;
    float pad_y = 0.f;
    int   frame_w = 0;
    int   frame_h = 0;
};

struct YoloParams {
    int   num_classes   = 80;
    int   num_proposals = 8400;
    float conf_threshold = 0.25f;
    float iou_threshold  = 0.45f;
};

/// output is channels-first [4+num_classes, num_proposals], no objectness. Returns
/// the count clamped to NVMM_META_MAX_OBJECTS and sets truncated on overflow.
uint32_t yolo_parse(const float *output, const YoloParams &p, const LetterboxInfo &lb,
                    NvmmDetObject *out_objects, bool *truncated);

const char *coco_label(int id);

}
