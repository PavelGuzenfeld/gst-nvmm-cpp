/// Preprocessor — NV12 NvBufSurface -> normalized NCHW float input tensor.
///
/// Honest "zero-copy" (no host round-trip): NvBufSurfTransform in GPU compute
/// mode, bound to the inference CUDA stream, resizes + letterboxes NV12 into an
/// RGBA surface; NPP then splits/converts/normalizes that into the planar f32
/// NCHW tensor TensorRT binds. All device-side, ordered on one stream.
#pragma once

#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

#include "yolo_parser.hpp"  // LetterboxInfo

namespace nvmm {

class Preprocessor {
public:
    /// net_w/net_h: model input size. color_rgb: RGB plane order (else BGR).
    /// scale: pixel multiplier (e.g. 1/255). Allocates the intermediate RGBA
    /// surface, sets the transform to GPU/stream, and pre-fills letterbox pad.
    bool configure(int net_w, int net_h, int frame_w, int frame_h,
                   bool color_rgb, float scale, cudaStream_t stream, std::string &err);

    /// Preprocess `src` into `d_input` (caller-owned, 3*net_w*net_h floats, NCHW)
    /// on the configured stream. Async — caller syncs after inference. Fills `lb`.
    bool run(NvBufSurface *src, float *d_input, LetterboxInfo &lb, std::string &err);

    bool configured() const { return rgba_ != nullptr; }
    ~Preprocessor();

private:
    int net_w_ = 0, net_h_ = 0;
    bool color_rgb_ = true;
    float scale_ = 1.f / 255.f;
    cudaStream_t stream_ = nullptr;

    LetterboxInfo lb_{};                 // constant geometry (frame size is fixed)
    NvBufSurfTransformRect dst_rect_{};

    NvBufSurface *rgba_ = nullptr;       // net RGBA, pitch-linear, CUDA memory
    uint8_t *planes_ = nullptr;          // 3 * net_w * net_h packed uint8
};

}  // namespace nvmm
