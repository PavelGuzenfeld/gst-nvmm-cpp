#pragma once

#include <cuda_runtime.h>

#include "image.hpp"
#include "low_texture_motion.hpp"

namespace nvmm {
namespace motion {

template <typename T>
struct DevicePlane {
    T *data = nullptr;
    int width = 0, height = 0;
    std::ptrdiff_t stride = 0;
};

class LowTextureMotionCuda {
public:
    LowTextureMotionCuda();
    ~LowTextureMotionCuda();
    LowTextureMotionCuda(const LowTextureMotionCuda &) = delete;
    LowTextureMotionCuda &operator=(const LowTextureMotionCuda &) = delete;

    bool run_device(DevicePlane<const uint8_t> cur, DevicePlane<const uint8_t> ref_a,
                    DevicePlane<const uint8_t> ref_b, const LowTextureMotionParams &p,
                    DevicePlane<float> out, cudaStream_t stream);

    bool run(img::View<const uint8_t> cur, img::View<const uint8_t> ref_a,
             img::View<const uint8_t> ref_b, const LowTextureMotionParams &p,
             img::Image<float> &out, cudaStream_t stream = nullptr);

    const char *last_error() const;

private:
    struct Impl;
    Impl *impl_;
};

}
}
