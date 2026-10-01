#pragma once

#include "nvmm_buffer.hpp"
#include "nvmm_types.hpp"

namespace nvmm {

class NvmmTransform {
public:
    static Result<void> transform(
        const NvmmBuffer& src,
        NvmmBuffer& dst,
        const TransformParams& params);

    static Result<void> scale(
        const NvmmBuffer& src,
        NvmmBuffer& dst);

    static Result<void> crop_and_scale(
        const NvmmBuffer& src,
        NvmmBuffer& dst,
        const CropRect& src_crop);

    static Result<void> convert(
        const NvmmBuffer& src,
        NvmmBuffer& dst);
};

}
