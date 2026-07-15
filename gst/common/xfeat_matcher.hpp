/// xfeat_matcher.hpp — XFeat + LightGlue feature matcher (TensorRT).
///
/// The OpenCV-free replacement for ORB + BFMatcher: extracts sparse learned
/// keypoints/descriptors from an NvBufSurface frame (XFeat CNN) and matches two
/// frames' features (LightGlue) into corresponding keypoint pairs. Downstream
/// analytics (GMC, independent-motion, drone gate) run on the matches via
/// common/xfeat_motion.hpp — no OpenCV anywhere.
///
/// Ports the validated driver in ../gst-nvmm-ostrack/gst/gstnvmmostrack.cpp
/// (init_xfeat/init_lightglue/xfeat_extract/register_point) onto this repo's
/// nvmm::TrtEngine + the raw NvBufSurfTransform (VIC) idiom used by
/// samurai_tracker.cpp. Built only where TensorRT + NvBufSurface are present
/// (compiled directly into each consuming element, like nvmminfer/trt_engine.cpp).
///
/// Preprocessing is a plain STRETCH full-frame -> 480x256 RGB /255 (NOT the YOLO
/// letterbox in nvmm::Preprocessor): letterbox bars produce false zero-displacement
/// matches that bias GMC toward zero, and padded input is off XFeat's training
/// distribution. Keypoints are returned in the 480x270 registration space
/// (x unchanged, y scaled by 270/256), matching the reference registration scale.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

#include "trt_engine.hpp"      // gst/nvmminfer — nvmm::TrtEngine
#include "xfeat_register.hpp"  // nvmm::xfeat::Pt2, filter_matches, ...
#include "xfeat_motion.hpp"    // nvmm::motion::MatchPair

namespace nvmm {

/// Extracted sparse features for one frame (registration-space coords).
struct XfeatFrame {
    std::vector<nvmm::xfeat::Pt2>     kpts;   // (x, y) in 480x270 space, score-sorted
    std::vector<std::array<float,64>> descs;  // L2-normalized 64-d descriptors
    bool empty() const { return kpts.empty(); }
};

class XfeatMatcher {
public:
    // XFeat CNN geometry + registration space (from the ostrack reference).
    static constexpr int    kXH = 256, kXW = 480;      // CNN input H,W
    static constexpr int    kXHC = 32, kXWC = 60;      // CNN output grid (H/8, W/8)
    static constexpr double kRW = 480.0, kRH = 270.0;  // registration space
    static constexpr double kRegScale = 0.25;          // registration space / full-res
    // Top-K keypoint cap: bounds LightGlue's O(N0*N1) cost and the sim buffer
    // (kTopK^2 floats). 1024 -> 4 MB sim per instance (vs 64 MB at the ref's 4096).
    static constexpr int    kTopK = 1024;

    XfeatMatcher() = default;
    ~XfeatMatcher();
    XfeatMatcher(const XfeatMatcher&) = delete;
    XfeatMatcher& operator=(const XfeatMatcher&) = delete;

    /// Load `xfeat.engine` + `lightglue.engine` from `engine_dir`, allocate device
    /// buffers and the RGBA scratch surface. False + `err` on failure.
    bool init(const std::string& engine_dir, std::string& err);

    /// VIC stretch full-frame -> 480x256 RGB/255 -> XFeat CNN -> sparse extraction.
    /// Returns features in 480x270 registration space. False + `err` on failure.
    bool extract(NvBufSurface* src, XfeatFrame& out, std::string& err);

    /// Match features of frame A (anchor) against frame B via LightGlue + host
    /// filter. Returns matched pairs with `idx` = A-keypoint index, `a` = A point,
    /// `b` = B point (all registration space). Empty vector if < the matcher's
    /// minimum (caller treats as "no verdict"). False + `err` only on a TRT error.
    bool match(const XfeatFrame& a, const XfeatFrame& b,
               std::vector<nvmm::motion::MatchPair>& out, std::string& err);

private:
    void free_buffers();

    std::unique_ptr<TrtEngine> xf_;   // XFeat CNN
    std::unique_ptr<TrtEngine> lg_;   // LightGlue matcher (dynamic-N)

    // XFeat device IO (static shapes).
    void* d_img_   = nullptr;         // image     (1,3,256,480)
    void* d_feats_ = nullptr;         // feats     (64,32,60)
    void* d_kpts_  = nullptr;         // keypoints (65,32,60)
    void* d_heat_  = nullptr;         // heatmap   (1,32,60)

    // LightGlue device IO (allocated to kTopK; shapes set per match()).
    void* d_d0_ = nullptr; void* d_d1_ = nullptr;
    void* d_k0_ = nullptr; void* d_k1_ = nullptr;
    void* d_sim_ = nullptr; void* d_z0_ = nullptr; void* d_z1_ = nullptr;

    NvBufSurface* rgba_ = nullptr;    // 480x256 RGBA VIC stretch destination
    cudaStream_t  stream_ = nullptr;
};

}  // namespace nvmm
