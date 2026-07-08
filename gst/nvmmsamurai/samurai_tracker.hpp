/// SamuraiTracker — C++/TensorRT port of the SAMURAI (SAM2.1) visual tracker.
///
/// Owns the five TensorRT engines (image encoder, prompt encoder, mask decoder,
/// memory encoder, memory attention) on one CUDA stream plus the learned
/// out-of-engine constants, and runs the per-frame track on an NV12 NvBufSurface.
/// B=1 (single target). The GStreamer element (gstnvmmsamurai.cpp) is a thin
/// shell around this; the heavy lifting (RoiCropper, MemoryBank, SamuraiSelector,
/// KalmanBox, MaskToBox) lives here and in its sibling units.
///
/// Engine I/O shapes (validated in Phase A, all FP16/TRT 10.3):
///   image_encoder  : input 1x3x512x512 -> out4 1x256x128x128 (feat_s0 raw),
///                    out5 1x256x64x64 (feat_s1 raw), out6 1x256x32x32 (image_embed)
///   prompt_encoder : coords 1x2x2 -> sparse 1x3x256, dense 1x256x32x32
///   mask_decoder   : image_embed, image_pe, sparse(1xNpx256), dense,
///                    feat_s0 1x256x128x128, feat_s1 1x256x64x64
///                    -> masks 1x4x128x128, ious 1x4, tokens 1x4x256, obj_score 1x1
///   memory_encoder : pix_feat 1x256x32x32, mask 1x1x512x512
///                    -> maskmem_feat 1x64x32x32, maskmem_pos 1x64x32x32
///   memory_attention: curr 1024x1x256, memory 7232x1x64, curr_pos, memory_pos
///                    -> attn 1024x1x256   (STATIC 7232; cold-start padded by MemoryBank)
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <cuda_runtime.h>
#include <nvbufsurface.h>

#include "gmc_backend.hpp"  // GmcBackend enum (camera-motion compensation backend)

namespace nvmm {

class TrtEngine;  // gst/nvmminfer/trt_engine.hpp

/// Axis-aligned box in frame pixel coords.
struct TrackBox {
    float left = 0.f, top = 0.f, width = 0.f, height = 0.f;
    float score = 0.f;       // SAM object-score logit (model box) or KF score
    bool  valid = false;
};

/// One frame's tracker result (mirrors the fields of GstNvmmTrackMeta).
struct TrackResult {
    TrackBox box;            // chosen/selected box, frame coords
    TrackBox kf_box;         // Kalman-predicted box, frame coords
    bool     is_kf_only = false;
    uint32_t stable_frames = 0;
    uint64_t target_id = 0;
};

struct SamuraiConfig {
    std::string engine_dir;          // dir holding the 5 *.engine files
    std::string consts_file;         // samurai_consts (.npz/.bin)
    int   crop_size = 512;           // encoder input (square)
    int   max_kf = 2;                // max consecutive KF-only frames
    float kf_score_weight = 0.25f;   // SAMURAI weighted = 0.25*kf_iou + 0.75*iou
    int   stable_frames_threshold = 10;
    float iou_threshold = 0.5f;      // min selected-candidate IoU to accept a KF update
    float kf_min_area = 25.f;        // min KF box area (px^2) to accept a KF update
    int   target_class = 0;          // YOLO class to seed from
    bool  gmc = false;               // camera-motion compensation (handheld clips)
    GmcBackend gmc_backend = GmcBackend::Ncc;  // GMC estimator; auto resolves at init
};

class SamuraiTracker {
public:
    SamuraiTracker();
    ~SamuraiTracker();
    SamuraiTracker(const SamuraiTracker &) = delete;
    SamuraiTracker &operator=(const SamuraiTracker &) = delete;

    /// Load the 5 engines + constants and create the CUDA stream.
    /// Returns false and fills `err` on failure.
    bool init(const SamuraiConfig &cfg, std::string &err);

    /// Seed (or re-seed) the tracker from a box (frame pixel coords) on `frame`.
    /// Runs encoder->prompt->decoder->memory-encoder once to write the
    /// conditioning frame. Returns false on failure.
    bool seed(NvBufSurface *frame, const TrackBox &box, std::string &err);

    /// Track on the next `frame`. If `kf_only`, runs no engines (KF predict only).
    /// Fills `out`. Returns false on a hard failure (engine error).
    bool track(NvBufSurface *frame, bool kf_only, TrackResult &out, std::string &err);

    bool seeded() const { return seeded_; }
    const SamuraiConfig &config() const { return cfg_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    SamuraiConfig cfg_;
    bool seeded_ = false;
};

}  // namespace nvmm
