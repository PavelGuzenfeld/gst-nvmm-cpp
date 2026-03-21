#include "nvmm_buffer.hpp"
#include "nvbufsurface_mock.h"

namespace nvmm {

NvmmBuffer::NvmmBuffer(NvBufSurface* surface) noexcept : surface_(surface) {}

NvmmBuffer::~NvmmBuffer() {
    if (surface_) {
        if (mapped_) {
            NvBufSurfaceUnMap(surface_, 0, -1);
        }
        NvBufSurfaceDestroy(surface_);
        surface_ = nullptr;
    }
}

NvmmBuffer::NvmmBuffer(NvmmBuffer&& other) noexcept
    : surface_(other.surface_), mapped_(other.mapped_) {
    other.surface_ = nullptr;
    other.mapped_ = false;
}

NvmmBuffer& NvmmBuffer::operator=(NvmmBuffer&& other) noexcept {
    if (this != &other) {
        if (surface_) {
            if (mapped_) NvBufSurfaceUnMap(surface_, 0, -1);
            NvBufSurfaceDestroy(surface_);
        }
        surface_ = other.surface_;
        mapped_ = other.mapped_;
        other.surface_ = nullptr;
        other.mapped_ = false;
    }
    return *this;
}

Result<NvmmBuffer> NvmmBuffer::create(const SurfaceParams& params) {
    if (params.width == 0 || params.height == 0) {
        return NvmmError{ErrorCode::kInvalidParam, "width and height must be > 0"};
    }

    NvBufSurfaceCreateParams create_params{};
    create_params.width = params.width;
    create_params.height = params.height;
    create_params.colorFormat = static_cast<NvBufSurfaceColorFormat>(params.color_format);
    create_params.memType = static_cast<NvBufSurfaceMemType>(params.mem_type);
    create_params.size = 0;
    create_params.layout = 0;
    create_params.isContiguous = 1;

    NvBufSurface* surface = nullptr;
    int ret = NvBufSurfaceCreate(&surface, params.num_surfaces, &create_params);
    if (ret != 0 || !surface) {
        return NvmmError{ErrorCode::kSurfaceCreateFailed,
                         "NvBufSurfaceCreate returned " + std::to_string(ret)};
    }

    return NvmmBuffer{surface};
}

Result<NvmmBuffer> NvmmBuffer::from_fd(int dmabuf_fd) {
    void* surf_ptr = nullptr;
    int ret = NvBufSurfaceFromFd(dmabuf_fd, &surf_ptr);
    if (ret != 0 || !surf_ptr) {
        return NvmmError{ErrorCode::kDmaBufFailed, "NvBufSurfaceFromFd failed"};
    }
    return NvmmBuffer{static_cast<NvBufSurface*>(surf_ptr)};
}

Result<ByteSpan> NvmmBuffer::map_read(uint32_t plane) {
    if (!surface_) {
        return NvmmError{ErrorCode::kInvalidParam, "null surface"};
    }

    int ret = NvBufSurfaceMap(surface_, 0, static_cast<int>(plane),
                              0 /* NVBUF_MAP_READ */);
    if (ret != 0) {
        return NvmmError{ErrorCode::kMapFailed, "NvBufSurfaceMap read failed"};
    }
    mapped_ = true;

    auto* addr = static_cast<uint8_t*>(surface_->surfaceList[0].mappedAddr);
    if (plane < surface_->surfaceList[0].planeParams.num_planes) {
        auto& pp = surface_->surfaceList[0].planeParams.planeParams[plane];
        return ByteSpan{addr + pp.offset, pp.psize};
    }
    return ByteSpan{addr, surface_->surfaceList[0].dataSize};
}

Result<ByteSpan> NvmmBuffer::map_write(uint32_t plane) {
    if (!surface_) {
        return NvmmError{ErrorCode::kInvalidParam, "null surface"};
    }

    int ret = NvBufSurfaceMap(surface_, 0, static_cast<int>(plane),
                              1 /* NVBUF_MAP_READ_WRITE */);
    if (ret != 0) {
        return NvmmError{ErrorCode::kMapFailed, "NvBufSurfaceMap write failed"};
    }
    mapped_ = true;

    auto* addr = static_cast<uint8_t*>(surface_->surfaceList[0].mappedAddr);
    if (plane < surface_->surfaceList[0].planeParams.num_planes) {
        auto& pp = surface_->surfaceList[0].planeParams.planeParams[plane];
        return ByteSpan{addr + pp.offset, pp.psize};
    }
    return ByteSpan{addr, surface_->surfaceList[0].dataSize};
}

Result<void> NvmmBuffer::unmap() {
    if (!surface_ || !mapped_) {
        return Result<void>{};
    }

    int ret = NvBufSurfaceUnMap(surface_, 0, -1);
    if (ret != 0) {
        return NvmmError{ErrorCode::kUnmapFailed, "NvBufSurfaceUnMap failed"};
    }
    mapped_ = false;
    return Result<void>{};
}

Result<int> NvmmBuffer::export_fd() const {
    if (!surface_) {
        return NvmmError{ErrorCode::kInvalidParam, "null surface"};
    }

    int fd = -1;
    int ret = NvBufSurfaceGetFd(surface_, 0, &fd);
    if (ret != 0) {
        return NvmmError{ErrorCode::kDmaBufFailed, "NvBufSurfaceGetFd failed"};
    }
    return fd;
}

uint32_t NvmmBuffer::width() const noexcept {
    return surface_ ? surface_->surfaceList[0].width : 0;
}

uint32_t NvmmBuffer::height() const noexcept {
    return surface_ ? surface_->surfaceList[0].height : 0;
}

ColorFormat NvmmBuffer::format() const noexcept {
    return surface_ ? static_cast<ColorFormat>(surface_->surfaceList[0].colorFormat)
                    : ColorFormat::kNV12;
}

MemoryType NvmmBuffer::mem_type() const noexcept {
    return surface_ ? static_cast<MemoryType>(surface_->memType)
                    : MemoryType::kDefault;
}

uint32_t NvmmBuffer::num_planes() const noexcept {
    return surface_ ? surface_->surfaceList[0].planeParams.num_planes : 0;
}

PlaneInfo NvmmBuffer::plane_info(uint32_t plane) const noexcept {
    if (!surface_ || plane >= surface_->surfaceList[0].planeParams.num_planes) {
        return {};
    }
    auto& pp = surface_->surfaceList[0].planeParams.planeParams[plane];
    return PlaneInfo{pp.width, pp.height, pp.pitch, pp.offset, pp.psize, pp.bytesPerPix};
}

uint32_t NvmmBuffer::data_size() const noexcept {
    return surface_ ? surface_->surfaceList[0].dataSize : 0;
}

}  // namespace nvmm
