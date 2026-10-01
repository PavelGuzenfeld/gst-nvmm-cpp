#pragma once

#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <npp.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

namespace nvmm {

class RoiPreprocessor {
public:
    bool configure(int net_w, int net_h, bool color_rgb, float scale,
                   const float *offsets, const float *std_values,
                   cudaStream_t stream, std::string &err);

    bool run(NvBufSurface *src, float left, float top, float width, float height,
             float *d_input, std::string &err);

    bool configured() const { return rgba_ != nullptr; }
    ~RoiPreprocessor();

private:
    int net_w_ = 0, net_h_ = 0;
    bool color_rgb_ = true;
    float scale_ = 1.f / 255.f;
    bool has_offsets_ = false, has_std_ = false;
    float offsets_[3] = {0.f, 0.f, 0.f};
    float std_[3] = {1.f, 1.f, 1.f};
    cudaStream_t stream_ = nullptr;
    NppStreamContext npp_ctx_{};

    NvBufSurface *rgba_ = nullptr;
    cudaGraphicsResource_t egl_res_ = nullptr;
    uint8_t *rgba_lin_ = nullptr;
    uint8_t *planes_ = nullptr;
};

}
