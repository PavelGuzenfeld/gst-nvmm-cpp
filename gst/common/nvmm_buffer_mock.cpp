/// NvmmBuffer implementation using mock NvBufSurface API.
/// Struct layout matches real API — same code works for both.
#include "nvmm_buffer.hpp"
#include "nvbufsurface_mock.h"

#include <string>

namespace nvmm {

namespace {

NvBufSurfaceColorFormat to_nv_format(ColorFormat fmt) {
    switch (fmt) {
        case ColorFormat::kNV12:  return NVBUF_COLOR_FORMAT_NV12;
        case ColorFormat::kRGBA:  return NVBUF_COLOR_FORMAT_RGBA;
        case ColorFormat::kBGRA:  return NVBUF_COLOR_FORMAT_BGRA;
        case ColorFormat::kI420:  return NVBUF_COLOR_FORMAT_YUV420;
        case ColorFormat::kNV21:  return NVBUF_COLOR_FORMAT_NV21;
        case ColorFormat::kGRAY8: return NVBUF_COLOR_FORMAT_GRAY8;
    }
    return NVBUF_COLOR_FORMAT_NV12;
}

ColorFormat from_nv_format(NvBufSurfaceColorFormat fmt) {
    switch (fmt) {
        case NVBUF_COLOR_FORMAT_NV12:    return ColorFormat::kNV12;
        case NVBUF_COLOR_FORMAT_RGBA:    return ColorFormat::kRGBA;
        case NVBUF_COLOR_FORMAT_BGRA:    return ColorFormat::kBGRA;
        case NVBUF_COLOR_FORMAT_YUV420:  return ColorFormat::kI420;
        case NVBUF_COLOR_FORMAT_NV21:    return ColorFormat::kNV21;
        case NVBUF_COLOR_FORMAT_GRAY8:   return ColorFormat::kGRAY8;
        default:                         return ColorFormat::kNV12;
    }
}

NvBufSurfaceMemType to_nv_memtype(MemoryType mt) {
    switch (mt) {
        case MemoryType::kDefault:      return NVBUF_MEM_DEFAULT;
        case MemoryType::kCudaDevice:   return NVBUF_MEM_CUDA_DEVICE;
        case MemoryType::kCudaPinned:   return NVBUF_MEM_CUDA_PINNED;
        case MemoryType::kCudaUnified:  return NVBUF_MEM_CUDA_UNIFIED;
        case MemoryType::kSurfaceArray: return NVBUF_MEM_SURFACE_ARRAY;
        case MemoryType::kHandle:       return NVBUF_MEM_HANDLE;
        case MemoryType::kSystemHeap:   return NVBUF_MEM_SYSTEM;
    }
    return NVBUF_MEM_DEFAULT;
}

MemoryType from_nv_memtype(NvBufSurfaceMemType mt) {
    switch (mt) {
        case NVBUF_MEM_DEFAULT:       return MemoryType::kDefault;
        case NVBUF_MEM_CUDA_DEVICE:   return MemoryType::kCudaDevice;
        case NVBUF_MEM_CUDA_PINNED:   return MemoryType::kCudaPinned;
        case NVBUF_MEM_CUDA_UNIFIED:  return MemoryType::kCudaUnified;
        case NVBUF_MEM_SURFACE_ARRAY: return MemoryType::kSurfaceArray;
        case NVBUF_MEM_HANDLE:        return MemoryType::kHandle;
        case NVBUF_MEM_SYSTEM:        return MemoryType::kSystemHeap;
    }
    return MemoryType::kDefault;
}

}  // namespace

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
    create_params.colorFormat = to_nv_format(params.color_format);
    create_params.memType = to_nv_memtype(params.mem_type);
    create_params.size = 0;
    create_params.layout = NVBUF_LAYOUT_PITCH;
    create_params.isContiguous = 1;

    NvBufSurface* surface = nullptr;
    int ret = NvBufSurfaceCreate(&surface, params.num_surfaces, &create_params);
    if (ret != 0 || !surface) {
        return NvmmError{ErrorCode::kSurfaceCreateFailed,
                         "NvBufSurfaceCreate returned " + std::to_string(ret)};
    }

    surface->numFilled = surface->batchSize;
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
                              NVBUF_MAP_READ);
    if (ret != 0) {
        return NvmmError{ErrorCode::kMapFailed, "NvBufSurfaceMap read failed"};
    }
    mapped_ = true;

    NvBufSurfaceSyncForCpu(surface_, 0, static_cast<int>(plane));

    auto& pp = surface_->surfaceList[0].planeParams;
    if (plane < pp.num_planes) {
        auto* addr = static_cast<uint8_t*>(surface_->surfaceList[0].mappedAddr.addr[plane]);
        return ByteSpan(addr, pp.psize[plane]);
    }
    auto* addr = static_cast<uint8_t*>(surface_->surfaceList[0].mappedAddr.addr[0]);
    return ByteSpan(addr, surface_->surfaceList[0].dataSize);
}

Result<ByteSpan> NvmmBuffer::map_write(uint32_t plane) {
    if (!surface_) {
        return NvmmError{ErrorCode::kInvalidParam, "null surface"};
    }

    int ret = NvBufSurfaceMap(surface_, 0, static_cast<int>(plane),
                              NVBUF_MAP_READ_WRITE);
    if (ret != 0) {
        return NvmmError{ErrorCode::kMapFailed, "NvBufSurfaceMap write failed"};
    }
    mapped_ = true;

    auto& pp = surface_->surfaceList[0].planeParams;
    if (plane < pp.num_planes) {
        auto* addr = static_cast<uint8_t*>(surface_->surfaceList[0].mappedAddr.addr[plane]);
        return ByteSpan(addr, pp.psize[plane]);
    }
    auto* addr = static_cast<uint8_t*>(surface_->surfaceList[0].mappedAddr.addr[0]);
    return ByteSpan(addr, surface_->surfaceList[0].dataSize);
}

Result<void> NvmmBuffer::unmap() {
    if (!surface_ || !mapped_) {
        return Result<void>{};
    }

    auto& pp = surface_->surfaceList[0].planeParams;
    for (uint32_t p = 0; p < pp.num_planes; p++) {
        NvBufSurfaceSyncForDevice(surface_, 0, static_cast<int>(p));
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

    int fd = static_cast<int>(surface_->surfaceList[0].bufferDesc);
    if (fd < 0) {
        return NvmmError{ErrorCode::kDmaBufFailed, "no DMA-buf fd available"};
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
    return surface_ ? from_nv_format(surface_->surfaceList[0].colorFormat)
                    : ColorFormat::kNV12;
}

MemoryType NvmmBuffer::mem_type() const noexcept {
    return surface_ ? from_nv_memtype(surface_->memType)
                    : MemoryType::kDefault;
}

uint32_t NvmmBuffer::num_planes() const noexcept {
    return surface_ ? surface_->surfaceList[0].planeParams.num_planes : 0;
}

PlaneInfo NvmmBuffer::plane_info(uint32_t plane) const noexcept {
    if (!surface_ || plane >= surface_->surfaceList[0].planeParams.num_planes) {
        return {};
    }
    auto& pp = surface_->surfaceList[0].planeParams;
    PlaneInfo info;
    info.width = pp.width[plane];
    info.height = pp.height[plane];
    info.pitch = pp.pitch[plane];
    info.offset = pp.offset[plane];
    info.size = pp.psize[plane];
    info.bytes_per_pixel = pp.bytesPerPix[plane];
    return info;
}

uint32_t NvmmBuffer::data_size() const noexcept {
    return surface_ ? surface_->surfaceList[0].dataSize : 0;
}

}  // namespace nvmm
