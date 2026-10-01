#pragma once

#include "nvmm_types.hpp"

struct NvBufSurface;

namespace nvmm {

class NvmmBuffer {
public:
    static Result<NvmmBuffer> create(const SurfaceParams& params);

    explicit NvmmBuffer(NvBufSurface* surface) noexcept;

    static Result<NvmmBuffer> from_fd(int dmabuf_fd);

    ~NvmmBuffer();

    NvmmBuffer(NvmmBuffer&& other) noexcept;
    NvmmBuffer& operator=(NvmmBuffer&& other) noexcept;
    NvmmBuffer(const NvmmBuffer&) = delete;
    NvmmBuffer& operator=(const NvmmBuffer&) = delete;

    Result<ByteSpan> map_read(uint32_t plane = 0);

    Result<ByteSpan> map_write(uint32_t plane = 0);

    Result<void> unmap();

    Result<int> export_fd() const;

    NvBufSurface* raw() const noexcept { return surface_; }

    /// The caller now owns the surface and must NvBufSurfaceDestroy it; use
    /// this when wrapping a borrowed surface.
    NvBufSurface* release() noexcept {
        NvBufSurface* s = surface_;
        surface_ = nullptr;
        mapped_ = false;
        return s;
    }

    uint32_t width() const noexcept;
    uint32_t height() const noexcept;
    ColorFormat format() const noexcept;
    MemoryType mem_type() const noexcept;
    uint32_t num_planes() const noexcept;
    PlaneInfo plane_info(uint32_t plane) const noexcept;
    uint32_t data_size() const noexcept;

    bool valid() const noexcept { return surface_ != nullptr; }
    explicit operator bool() const noexcept { return valid(); }

private:
    NvBufSurface* surface_ = nullptr;
    bool mapped_ = false;
    int mapped_plane_ = -1;
};

}
