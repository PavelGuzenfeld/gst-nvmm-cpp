#pragma once

#include <cuda_runtime.h>

#include "image.hpp"
#include "low_texture_motion.hpp"

namespace nvmm {
namespace motion {

/// Device counterpart of img::View: `stride` in elements of T (a mapped NvBufSurface
/// pitch is bytes, divide by sizeof(T)). A separate type so host code cannot
/// dereference a device pointer through View::at()/row().
template <typename T>
struct DevicePlane {
    T *data = nullptr;
    int width = 0, height = 0;
    std::ptrdiff_t stride = 0;
};

/// Owns device scratch only, never the input/output planes. Not thread-safe: one
/// instance per stream. False on any CUDA error; callers fall back to the host path.
class LowTextureMotionCuda {
public:
    LowTextureMotionCuda();
    ~LowTextureMotionCuda();
    LowTextureMotionCuda(const LowTextureMotionCuda &) = delete;
    LowTextureMotionCuda &operator=(const LowTextureMotionCuda &) = delete;

    /// Enqueued on `stream` and not synchronised; the caller owns the sync point.
    bool run_device(DevicePlane<const uint8_t> cur, DevicePlane<const uint8_t> ref_a,
                    DevicePlane<const uint8_t> ref_b, const LowTextureMotionParams &p,
                    DevicePlane<float> out, cudaStream_t stream);

    /// Uploads, runs on `stream` (default stream when null), downloads, synchronises.
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
