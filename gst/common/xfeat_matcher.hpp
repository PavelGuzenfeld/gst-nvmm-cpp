#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

#include "trt_engine.hpp"
#include "xfeat_register.hpp"
#include "xfeat_motion.hpp"

namespace nvmm {

struct XfeatFrame {
    std::vector<nvmm::xfeat::Pt2>     kpts;
    std::vector<std::array<float,64>> descs;
    bool empty() const { return kpts.empty(); }
};

class XfeatMatcher {
public:
    static constexpr int    kXH = 256, kXW = 480;
    static constexpr int    kXHC = 32, kXWC = 60;
    static constexpr double kRW = 480.0, kRH = 270.0;
    static constexpr double kRegScale = 0.25;
    static constexpr int    kTopK = 1024;

    XfeatMatcher() = default;
    ~XfeatMatcher();
    XfeatMatcher(const XfeatMatcher&) = delete;
    XfeatMatcher& operator=(const XfeatMatcher&) = delete;

    bool init(const std::string& engine_dir, std::string& err);

    bool extract(NvBufSurface* src, XfeatFrame& out, std::string& err);

    bool match(const XfeatFrame& a, const XfeatFrame& b,
               std::vector<nvmm::motion::MatchPair>& out, std::string& err);

private:
    void free_buffers();

    std::unique_ptr<TrtEngine> xf_;
    std::unique_ptr<TrtEngine> lg_;

    void* d_img_   = nullptr;
    void* d_feats_ = nullptr;
    void* d_kpts_  = nullptr;
    void* d_heat_  = nullptr;

    void* d_d0_ = nullptr; void* d_d1_ = nullptr;
    void* d_k0_ = nullptr; void* d_k1_ = nullptr;
    void* d_sim_ = nullptr; void* d_z0_ = nullptr; void* d_z1_ = nullptr;

    NvBufSurface* rgba_ = nullptr;
    cudaStream_t  stream_ = nullptr;
};

}
