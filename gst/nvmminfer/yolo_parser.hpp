/// YOLO11 / YOLOv8 detection-head parser (pure host, no CUDA).
///
/// Decodes the single output tensor `output0` of shape [1, 84, 8400] (channels-
/// first: 84 = 4 box + 80 COCO class scores, across 8400 proposals; no separate
/// objectness) into detections, applies per-class NMS, and maps boxes from the
/// letterboxed network space back to the original frame's pixel space.
#pragma once

#include <cstdint>
#include <vector>

#include "shm_protocol.h"  // NvmmDetObject, NVMM_META_*

namespace nvmm {

/// Letterbox geometry used to map network-space boxes back to frame pixels:
/// frame_x = (net_x - pad_x) / scale.
struct LetterboxInfo {
    float scale = 1.f;   // net = round(frame * scale)
    float pad_x = 0.f;   // left padding added in network space
    float pad_y = 0.f;   // top padding
    int   frame_w = 0;   // original frame size (the det_meta coordinate space)
    int   frame_h = 0;
};

struct YoloParams {
    int   num_classes   = 80;
    int   num_proposals = 8400;
    float conf_threshold = 0.25f;
    float iou_threshold  = 0.45f;
};

/// Parse `output` (num_proposals*(4+num_classes) floats, channels-first) into
/// `out_objects` (already-NMS'd, mapped to frame pixels). Returns the count
/// (clamped to NVMM_META_MAX_OBJECTS); sets `truncated` if it overflowed.
uint32_t yolo_parse(const float *output, const YoloParams &p, const LetterboxInfo &lb,
                    NvmmDetObject *out_objects, bool *truncated);

/// COCO-80 class label for `id` ("" if out of range).
const char *coco_label(int id);

}  // namespace nvmm
