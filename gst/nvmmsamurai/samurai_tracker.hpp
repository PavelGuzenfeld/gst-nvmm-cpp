#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <cuda_runtime.h>
#include <nvbufsurface.h>

#include "gmc_backend.hpp"

namespace nvmm {

class TrtEngine;

struct TrackBox {
    float left = 0.f, top = 0.f, width = 0.f, height = 0.f;
    float score = 0.f;
    bool  valid = false;
};

struct TrackResult {
    TrackBox box;
    TrackBox kf_box;
    bool     is_kf_only = false;
    uint32_t stable_frames = 0;
    uint64_t target_id = 0;
};

struct SamuraiConfig {
    std::string engine_dir;
    std::string consts_file;
    int   crop_size = 512;
    int   max_kf = 2;
    float kf_score_weight = 0.25f;
    int   stable_frames_threshold = 10;
    float iou_threshold = 0.5f;
    float kf_min_area = 25.f;
    int   target_class = 0;
    bool  gmc = false;
    GmcBackend gmc_backend = GmcBackend::Ncc;
};

class SamuraiTracker {
public:
    SamuraiTracker();
    ~SamuraiTracker();
    SamuraiTracker(const SamuraiTracker &) = delete;
    SamuraiTracker &operator=(const SamuraiTracker &) = delete;

    bool init(const SamuraiConfig &cfg, std::string &err);

    bool seed(NvBufSurface *frame, const TrackBox &box, std::string &err);

    bool track(NvBufSurface *frame, bool kf_only, TrackResult &out, std::string &err);

    bool seeded() const { return seeded_; }
    const SamuraiConfig &config() const { return cfg_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    SamuraiConfig cfg_;
    bool seeded_ = false;
};

}
