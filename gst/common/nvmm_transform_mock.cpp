#include "nvmm_transform.hpp"
#include "nvbufsurface_mock.h"

namespace nvmm {

Result<void> NvmmTransform::transform(
    const NvmmBuffer& src, NvmmBuffer& dst, const TransformParams& params) {
    if (!src.raw() || !dst.raw()) {
        return NvmmError{ErrorCode::kInvalidParam, "null surface in transform"};
    }

    NvBufSurfTransformParams xform{};
    xform.flip = static_cast<NvBufSurfTransform_Flip>(params.flip);

    NvBufSurfTransformRect src_rect{};
    NvBufSurfTransformRect dst_rect{};

    if (params.src_crop.is_valid()) {
        src_rect.top = params.src_crop.y;
        src_rect.left = params.src_crop.x;
        src_rect.width = params.src_crop.width;
        src_rect.height = params.src_crop.height;
        xform.src_rect = &src_rect;
        xform.transform_flag |= 1; // NVBUFSURF_TRANSFORM_CROP_SRC
    }

    if (params.dst_crop.is_valid()) {
        dst_rect.top = params.dst_crop.y;
        dst_rect.left = params.dst_crop.x;
        dst_rect.width = params.dst_crop.width;
        dst_rect.height = params.dst_crop.height;
        xform.dst_rect = &dst_rect;
        xform.transform_flag |= 2; // NVBUFSURF_TRANSFORM_CROP_DST
    }

    if (params.flip != FlipMethod::kNone) {
        xform.transform_flag |= 4; // NVBUFSURF_TRANSFORM_FLIP
    }

    int ret = NvBufSurfTransform(src.raw(), dst.raw(), &xform);
    if (ret != 0) {
        return NvmmError{ErrorCode::kTransformFailed,
                         "NvBufSurfTransform returned " + std::to_string(ret)};
    }
    return Result<void>{};
}

Result<void> NvmmTransform::scale(const NvmmBuffer& src, NvmmBuffer& dst) {
    return transform(src, dst, TransformParams{});
}

Result<void> NvmmTransform::crop_and_scale(
    const NvmmBuffer& src, NvmmBuffer& dst, const CropRect& src_crop) {
    TransformParams params;
    params.src_crop = src_crop;
    return transform(src, dst, params);
}

Result<void> NvmmTransform::convert(const NvmmBuffer& src, NvmmBuffer& dst) {
    return transform(src, dst, TransformParams{});
}

}  // namespace nvmm
