#include "yolo_parser.hpp"
#include "test_harness.h"

#include <cmath>
#include <cstring>
#include <vector>

using nvmm::LetterboxInfo;
using nvmm::YoloParams;

namespace {
constexpr int N = 8400;
constexpr int C = 80;

void set_prop(std::vector<float> &o, int i, float cx, float cy, float w, float h,
              int cls, float score) {
    o[0 * N + i] = cx; o[1 * N + i] = cy; o[2 * N + i] = w; o[3 * N + i] = h;
    o[(4 + cls) * N + i] = score;
}

std::vector<float> blank() { return std::vector<float>((size_t)(4 + C) * N, 0.f); }
}

TEST(decode_single) {
    auto o = blank();
    set_prop(o, 0, 320, 320, 100, 200, 5, 0.9f);
    YoloParams p;
    LetterboxInfo lb{1.f, 0.f, 0.f, 640, 640};
    NvmmDetObject out[NVMM_META_MAX_OBJECTS];
    bool trunc = false;
    uint32_t n = nvmm::yolo_parse(o.data(), p, lb, out, &trunc);
    ASSERT_EQ(n, 1u);
    ASSERT_EQ(out[0].class_id, 5);
    ASSERT_TRUE(std::strcmp(out[0].label, "bus") == 0);
    ASSERT_NEAR(out[0].confidence, 0.9f, 1e-5);
    ASSERT_NEAR(out[0].left, 270.f, 0.5);
    ASSERT_NEAR(out[0].top, 220.f, 0.5);
    ASSERT_NEAR(out[0].width, 100.f, 0.5);
    ASSERT_NEAR(out[0].height, 200.f, 0.5);
    ASSERT_TRUE(!trunc);
}

TEST(letterbox_unmap) {
    auto o = blank();
    set_prop(o, 0, 400, 240, 100, 80, 2, 0.8f);
    YoloParams p;
    LetterboxInfo lb{0.5f, 80.f, 40.f, 4096, 4096};
    NvmmDetObject out[NVMM_META_MAX_OBJECTS];
    uint32_t n = nvmm::yolo_parse(o.data(), p, lb, out, nullptr);
    ASSERT_EQ(n, 1u);
    ASSERT_NEAR(out[0].left, 540.f, 0.5);
    ASSERT_NEAR(out[0].top, 320.f, 0.5);
    ASSERT_NEAR(out[0].width, 200.f, 0.5);
    ASSERT_NEAR(out[0].height, 160.f, 0.5);
}

TEST(conf_threshold) {
    auto o = blank();
    set_prop(o, 0, 320, 320, 100, 100, 5, 0.20f);
    YoloParams p;
    LetterboxInfo lb{1.f, 0.f, 0.f, 640, 640};
    NvmmDetObject out[NVMM_META_MAX_OBJECTS];
    ASSERT_EQ(nvmm::yolo_parse(o.data(), p, lb, out, nullptr), 0u);
}

TEST(nms_same_class_suppresses) {
    auto o = blank();
    set_prop(o, 0, 320, 320, 100, 200, 5, 0.90f);
    set_prop(o, 1, 325, 322, 100, 200, 5, 0.80f);
    YoloParams p;
    LetterboxInfo lb{1.f, 0.f, 0.f, 640, 640};
    NvmmDetObject out[NVMM_META_MAX_OBJECTS];
    uint32_t n = nvmm::yolo_parse(o.data(), p, lb, out, nullptr);
    ASSERT_EQ(n, 1u);
    ASSERT_NEAR(out[0].confidence, 0.90f, 1e-5);
}

TEST(nms_cross_class_keeps_both) {
    auto o = blank();
    set_prop(o, 0, 320, 320, 100, 200, 5, 0.90f);
    set_prop(o, 1, 322, 321, 100, 200, 2, 0.85f);
    YoloParams p;
    LetterboxInfo lb{1.f, 0.f, 0.f, 640, 640};
    NvmmDetObject out[NVMM_META_MAX_OBJECTS];
    ASSERT_EQ(nvmm::yolo_parse(o.data(), p, lb, out, nullptr), 2u);
}

TEST(truncation_flag) {
    auto o = blank();
    const int M = NVMM_META_MAX_OBJECTS + 40;
    for (int i = 0; i < M; i++) {
        float cx = (float)(i % 40) * 20 + 5, cy = (float)(i / 40) * 20 + 5;
        set_prop(o, i, cx, cy, 5, 5, i % C, 0.9f);
    }
    YoloParams p;
    LetterboxInfo lb{1.f, 0.f, 0.f, 4096, 4096};
    NvmmDetObject out[NVMM_META_MAX_OBJECTS];
    bool trunc = false;
    uint32_t n = nvmm::yolo_parse(o.data(), p, lb, out, &trunc);
    ASSERT_EQ(n, (uint32_t)NVMM_META_MAX_OBJECTS);
    ASSERT_TRUE(trunc);
}

int main() {
    printf("=== yolo_parser tests ===\n");
    return tests_failed > 0 ? 1 : 0;
}
