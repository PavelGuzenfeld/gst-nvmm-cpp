#pragma once

#include "nvmm_buffer.hpp"
#include "nvmm_types.hpp"

namespace nvmm {

/// Wraps NvBufSurfTransform — the Tegra VIC hardware engine for
/// crop, scale, and color format conversion within NVMM memory.
/// All operations are zero-copy (GPU/VIC-side, no CPU involvement).
class NvmmTransform {
public:
    /// Transform src buffer into dst buffer with the given parameters.
    /// Both buffers must be NVMM-allocated.
    static Result<void> transform(
        const NvmmBuffer& src,
        NvmmBuffer& dst,
        const TransformParams& params);

    /// Simple scale: resize src into dst (dst dimensions determine output size).
    static Result<void> scale(
        const NvmmBuffer& src,
        NvmmBuffer& dst);

    /// Crop and scale: crop a region from src, scale to fill dst.
    static Result<void> crop_and_scale(
        const NvmmBuffer& src,
        NvmmBuffer& dst,
        const CropRect& src_crop);

    /// Format convert: src format -> dst format (dst must be pre-allocated
    /// with desired format).
    static Result<void> convert(
        const NvmmBuffer& src,
        NvmmBuffer& dst);
};

}  // namespace nvmm
