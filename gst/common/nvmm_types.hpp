#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <variant>

namespace nvmm {

/// Error codes for NVMM operations
enum class ErrorCode : int {
    kSuccess = 0,
    kInvalidParam,
    kOutOfMemory,
    kNotSupported,
    kMapFailed,
    kUnmapFailed,
    kTransformFailed,
    kDmaBufFailed,
    kSurfaceCreateFailed,
    kSurfaceDestroyFailed,
};

/// Human-readable error category
class NvmmErrorCategory : public std::error_category {
public:
    [[nodiscard]] const char* name() const noexcept override {
        return "nvmm";
    }

    [[nodiscard]] std::string message(int ev) const override {
        switch (static_cast<ErrorCode>(ev)) {
            case ErrorCode::kSuccess:             return "success";
            case ErrorCode::kInvalidParam:        return "invalid parameter";
            case ErrorCode::kOutOfMemory:         return "out of memory";
            case ErrorCode::kNotSupported:        return "not supported";
            case ErrorCode::kMapFailed:           return "surface map failed";
            case ErrorCode::kUnmapFailed:         return "surface unmap failed";
            case ErrorCode::kTransformFailed:     return "surface transform failed";
            case ErrorCode::kDmaBufFailed:        return "DMA-buf operation failed";
            case ErrorCode::kSurfaceCreateFailed: return "surface create failed";
            case ErrorCode::kSurfaceDestroyFailed:return "surface destroy failed";
        }
        return "unknown nvmm error";
    }
};

inline const NvmmErrorCategory& nvmm_error_category() {
    static const NvmmErrorCategory instance;
    return instance;
}

inline std::error_code make_error_code(ErrorCode e) {
    return {static_cast<int>(e), nvmm_error_category()};
}

/// NvmmError wraps an error code with optional context
struct NvmmError {
    ErrorCode code;
    std::string detail;

    NvmmError(ErrorCode c, std::string d = {})
        : code(c), detail(std::move(d)) {}
};

/// C++17 Result type using std::variant (value | error).
/// Inspired by std::expected (C++23) but compatible with C++17.
template <typename T>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}  // NOLINT: implicit
    Result(NvmmError error) : data_(std::move(error)) {}  // NOLINT: implicit

    [[nodiscard]] bool has_value() const noexcept {
        return std::holds_alternative<T>(data_);
    }
    explicit operator bool() const noexcept { return has_value(); }

    [[nodiscard]] T& value() & { return std::get<T>(data_); }
    [[nodiscard]] const T& value() const& { return std::get<T>(data_); }
    [[nodiscard]] T&& value() && { return std::get<T>(std::move(data_)); }

    [[nodiscard]] NvmmError& error() & { return std::get<NvmmError>(data_); }
    [[nodiscard]] const NvmmError& error() const& { return std::get<NvmmError>(data_); }

    [[nodiscard]] T& operator*() & { return value(); }
    [[nodiscard]] const T& operator*() const& { return value(); }

private:
    std::variant<T, NvmmError> data_;
};

/// Specialization for void Result
template <>
class Result<void> {
public:
    Result() = default;
    Result(NvmmError error) : error_(std::move(error)) {}  // NOLINT: implicit

    [[nodiscard]] bool has_value() const noexcept { return !error_.has_value(); }
    explicit operator bool() const noexcept { return has_value(); }

    [[nodiscard]] const NvmmError& error() const { return error_.value(); }

private:
    std::optional<NvmmError> error_;
};

/// Lightweight non-owning view of contiguous bytes (C++17 replacement for std::span)
class ByteSpan {
public:
    ByteSpan() = default;
    ByteSpan(uint8_t* data, std::size_t size) : data_(data), size_(size) {}

    [[nodiscard]] uint8_t* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    uint8_t& operator[](std::size_t i) { return data_[i]; }
    const uint8_t& operator[](std::size_t i) const { return data_[i]; }

    [[nodiscard]] uint8_t* begin() const noexcept { return data_; }
    [[nodiscard]] uint8_t* end() const noexcept { return data_ + size_; }

private:
    uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
};

/// Memory types matching NvBufSurfaceMemType
enum class MemoryType : int {
    kDefault = 0,       // NVBUF_MEM_DEFAULT
    kCudaDevice = 1,    // NVBUF_MEM_CUDA_DEVICE
    kCudaPinned = 2,    // NVBUF_MEM_CUDA_PINNED
    kCudaUnified = 3,   // NVBUF_MEM_CUDA_UNIFIED
    kSurfaceArray = 4,  // NVBUF_MEM_SURFACE_ARRAY
    kHandle = 5,        // NVBUF_MEM_HANDLE
    kSystemHeap = 6,    // NVBUF_MEM_SYSTEM
};

/// Color formats matching NvBufSurfaceColorFormat (subset)
enum class ColorFormat : int {
    kNV12 = 0,
    kRGBA = 1,
    kBGRA = 2,
    kI420 = 3,
    kNV21 = 4,
    kGRAY8 = 5,
};

/// Plane info for a buffer surface
struct PlaneInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t pitch = 0;
    uint32_t offset = 0;
    uint32_t size = 0;
    uint32_t bytes_per_pixel = 0;
};

/// Parameters for creating NVMM surfaces
struct SurfaceParams {
    uint32_t width = 0;
    uint32_t height = 0;
    ColorFormat color_format = ColorFormat::kNV12;
    MemoryType mem_type = MemoryType::kSurfaceArray;
    uint32_t num_surfaces = 1;
};

/// Flip method for transforms
enum class FlipMethod : int {
    kNone = 0,
    kRotate90CW = 1,
    kRotate180 = 2,
    kRotate90CCW = 3,
    kFlipHorizontal = 4,
    kFlipUpperRightToLowerLeft = 5,
    kFlipVertical = 6,
    kFlipUpperLeftToLowerRight = 7,
};

/// Crop rectangle
struct CropRect {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;

    [[nodiscard]] bool is_valid() const {
        return width > 0 && height > 0;
    }
};

/// Transform parameters
struct TransformParams {
    CropRect src_crop;
    CropRect dst_crop;
    FlipMethod flip = FlipMethod::kNone;
};

}  // namespace nvmm
