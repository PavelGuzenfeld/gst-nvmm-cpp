#pragma once

#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

#include "yolo_parser.hpp"

namespace nvmm {

class Preprocessor {
public:
    bool configure(int net_w, int net_h, int frame_w, int frame_h,
                   bool color_rgb, float scale, cudaStream_t stream, std::string &err);

    bool run(NvBufSurface *src, float *d_input, LetterboxInfo &lb, std::string &err);

    bool configured() const { return rgba_ != nullptr; }
    ~Preprocessor();

private:
    int net_w_ = 0, net_h_ = 0;
    bool color_rgb_ = true;
    float scale_ = 1.f / 255.f;
    cudaStream_t stream_ = nullptr;

    LetterboxInfo lb_{};
    NvBufSurfTransformRect dst_rect_{};

    NvBufSurface *rgba_ = nullptr;
    cudaGraphicsResource_t egl_res_ = nullptr;
    uint8_t *rgba_lin_ = nullptr;
    uint8_t *planes_ = nullptr;
};

}
