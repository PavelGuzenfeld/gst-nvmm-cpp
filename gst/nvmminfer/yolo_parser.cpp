#include "yolo_parser.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace nvmm {

namespace {

const char *const kCoco80[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
};
constexpr int kCocoCount = (int)(sizeof(kCoco80) / sizeof(kCoco80[0]));

struct Det {
    float x1, y1, x2, y2;  // frame-pixel space
    float score;
    int   cls;
};

float iou(const Det &a, const Det &b) {
    const float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
    const float inter = iw * ih;
    const float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0.f ? inter / ua : 0.f;
}

}  // namespace

const char *coco_label(int id) {
    return (id >= 0 && id < kCocoCount) ? kCoco80[id] : "";
}

uint32_t yolo_parse(const float *output, const YoloParams &p, const LetterboxInfo &lb,
                    NvmmDetObject *out_objects, bool *truncated) {
    const int N = p.num_proposals;
    const int C = p.num_classes;
    const float inv_scale = lb.scale > 0.f ? 1.f / lb.scale : 1.f;

    std::vector<Det> cands;
    cands.reserve(256);

    for (int i = 0; i < N; i++) {
        // Best class for this proposal (channels-first: out[c*N + i]).
        int best = -1;
        float best_s = p.conf_threshold;
        for (int k = 0; k < C; k++) {
            const float s = output[(4 + k) * N + i];
            if (s > best_s) { best_s = s; best = k; }
        }
        if (best < 0) continue;

        const float cx = output[0 * N + i], cy = output[1 * N + i];
        const float w  = output[2 * N + i], h  = output[3 * N + i];

        // Network space -> frame pixels (undo letterbox), then clamp.
        float x1 = (cx - w * 0.5f - lb.pad_x) * inv_scale;
        float y1 = (cy - h * 0.5f - lb.pad_y) * inv_scale;
        float x2 = (cx + w * 0.5f - lb.pad_x) * inv_scale;
        float y2 = (cy + h * 0.5f - lb.pad_y) * inv_scale;
        x1 = std::min((float)lb.frame_w, std::max(0.f, x1));
        y1 = std::min((float)lb.frame_h, std::max(0.f, y1));
        x2 = std::min((float)lb.frame_w, std::max(0.f, x2));
        y2 = std::min((float)lb.frame_h, std::max(0.f, y2));
        if (x2 <= x1 || y2 <= y1) continue;

        cands.push_back(Det{x1, y1, x2, y2, best_s, best});
    }

    // Greedy per-class NMS: sort by score desc, suppress same-class overlaps.
    std::sort(cands.begin(), cands.end(),
              [](const Det &a, const Det &b) { return a.score > b.score; });

    std::vector<char> removed(cands.size(), 0);
    uint32_t kept = 0;
    bool over = false;

    for (size_t a = 0; a < cands.size(); a++) {
        if (removed[a]) continue;
        for (size_t b = a + 1; b < cands.size(); b++) {
            if (!removed[b] && cands[b].cls == cands[a].cls &&
                iou(cands[a], cands[b]) > p.iou_threshold)
                removed[b] = 1;
        }
        if (kept >= NVMM_META_MAX_OBJECTS) { over = true; break; }

        NvmmDetObject &o = out_objects[kept++];
        o.left = cands[a].x1;
        o.top = cands[a].y1;
        o.width = cands[a].x2 - cands[a].x1;
        o.height = cands[a].y2 - cands[a].y1;
        o.class_id = cands[a].cls;
        o.confidence = cands[a].score;
        o.tracker_id = 0;  // no tracker at the detector stage
        std::strncpy(o.label, coco_label(cands[a].cls), NVMM_META_LABEL_LEN - 1);
        o.label[NVMM_META_LABEL_LEN - 1] = '\0';
    }

    if (truncated) *truncated = over;
    return kept;
}

}  // namespace nvmm
