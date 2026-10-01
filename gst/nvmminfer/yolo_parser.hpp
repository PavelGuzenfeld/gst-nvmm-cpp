#pragma once

#include <cstdint>
#include <vector>

#include "shm_protocol.h"

namespace nvmm {

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

uint32_t yolo_parse(const float *output, const YoloParams &p, const LetterboxInfo &lb,
                    NvmmDetObject *out_objects, bool *truncated);

const char *coco_label(int id);

}
