#pragma once

#include "nvmm_types.hpp"

// Forward-declare the NVIDIA type
struct NvBufSurface;

namespace nvmm {

/// RAII wrapper for NvBufSurface.
/// Owns the surface and destroys it in the destructor.
/// Move-only — no copies.
class NvmmBuffer {
public:
    /// Create a new NVMM buffer with the given parameters.
    [[nodiscard]] static Result<NvmmBuffer> create(const SurfaceParams& params);

    /// Wrap an existing NvBufSurface (takes ownership).
    explicit NvmmBuffer(NvBufSurface* surface) noexcept;

    /// Wrap from a DMA-buf fd (imports the surface).
    [[nodiscard]] static Result<NvmmBuffer> from_fd(int dmabuf_fd);

    ~NvmmBuffer();

    // Move-only
    NvmmBuffer(NvmmBuffer&& other) noexcept;
    NvmmBuffer& operator=(NvmmBuffer&& other) noexcept;
    NvmmBuffer(const NvmmBuffer&) = delete;
    NvmmBuffer& operator=(const NvmmBuffer&) = delete;

    /// Map surface for CPU read access. Returns view of plane data.
    [[nodiscard]] Result<ByteSpan> map_read(uint32_t plane = 0);

    /// Map surface for CPU write access. Returns view of plane data.
    [[nodiscard]] Result<ByteSpan> map_write(uint32_t plane = 0);

    /// Unmap previously mapped surface.
    Result<void> unmap();

    /// Export as DMA-buf file descriptor for V4L2/display interop.
    [[nodiscard]] Result<int> export_fd() const;

    /// Get the raw NvBufSurface pointer (non-owning).
    [[nodiscard]] NvBufSurface* raw() const noexcept { return surface_; }

    /// Get surface info
    [[nodiscard]] uint32_t width() const noexcept;
    [[nodiscard]] uint32_t height() const noexcept;
    [[nodiscard]] ColorFormat format() const noexcept;
    [[nodiscard]] MemoryType mem_type() const noexcept;
    [[nodiscard]] uint32_t num_planes() const noexcept;
    [[nodiscard]] PlaneInfo plane_info(uint32_t plane) const noexcept;
    [[nodiscard]] uint32_t data_size() const noexcept;

    /// Check if valid
    [[nodiscard]] bool valid() const noexcept { return surface_ != nullptr; }
    explicit operator bool() const noexcept { return valid(); }

private:
    NvBufSurface* surface_ = nullptr;
    bool mapped_ = false;
};

}  // namespace nvmm
