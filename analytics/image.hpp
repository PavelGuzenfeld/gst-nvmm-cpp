/// Minimal 2D image buffer + strided view for the analytics components.
///
/// Replaces the cv::Mat surface these headers used to lean on, with just what
/// the fused implementations need: an owning single-channel buffer and a
/// non-owning strided view. The view is the public currency — a caller can wrap
/// any strided plane (e.g. a CPU-mapped NVMM luma plane) without a copy. No
/// elementwise-op methods on purpose: operations live inside the components'
/// fused passes, not on the buffer type.
///
/// Pure C++14, header-only, no dependencies.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace nvmm {
namespace img {

struct Rect {
    int x = 0, y = 0, w = 0, h = 0;
};

/// Non-owning view of a strided single-channel 2D buffer. `stride` is in
/// ELEMENTS (not bytes). Use View<const T> for read-only access.
template <typename T>
struct View {
    T *data = nullptr;
    int width = 0, height = 0;
    std::ptrdiff_t stride = 0;

    View() = default;
    View(T *d, int w, int h, std::ptrdiff_t s) : data(d), width(w), height(h), stride(s) {}

    bool empty() const { return data == nullptr || width <= 0 || height <= 0; }
    T *row(int y) const { return data + (std::ptrdiff_t)y * stride; }
    T &at(int y, int x) const { return row(y)[x]; }

    /// View<T> -> View<const T> (SFINAE'd away when T is already const).
    /// U must be pinned to T via is_same: without it, the conversion-operator
    /// template lets the compiler deduce U from the CONVERSION TARGET alone
    /// (e.g. U=float when converting a View<uint8_t> to View<const float> in a
    /// call's overload resolution), since a conversion operator has no function
    /// parameter tying U back to the class's actual T. That made View<uint8_t>
    /// spuriously "convertible" to View<const AnyType>, which under -std=c++20
    /// surfaced as an ambiguous overload between unrelated process(View<const
    /// uint8_t>) / process(View<const float>) call targets.
    template <typename U = T,
              typename std::enable_if<std::is_same<U, T>::value &&
                                      !std::is_const<U>::value, int>::type = 0>
    operator View<const U>() const { return View<const U>(data, width, height, stride); }

    View sub(const Rect &r) const {
        return View(data + (std::ptrdiff_t)r.y * stride + r.x, r.w, r.h, stride);
    }
};

/// Owning, densely-packed (stride == width) single-channel buffer.
template <typename T>
class Image {
public:
    Image() = default;
    Image(int w, int h, T fill = T()) : w_(w), h_(h), d_((size_t)w * h, fill) {}

    bool empty() const { return d_.empty(); }
    int width() const { return w_; }
    int height() const { return h_; }

    T *row(int y) { return d_.data() + (size_t)y * w_; }
    const T *row(int y) const { return d_.data() + (size_t)y * w_; }
    T &at(int y, int x) { return row(y)[x]; }
    const T &at(int y, int x) const { return row(y)[x]; }
    T *data() { return d_.data(); }
    const T *data() const { return d_.data(); }

    View<T> view() { return View<T>(d_.data(), w_, h_, w_); }
    View<const T> view() const { return View<const T>(d_.data(), w_, h_, w_); }
    operator View<T>() { return view(); }
    operator View<const T>() const { return view(); }

    void release() { w_ = h_ = 0; d_.clear(); d_.shrink_to_fit(); }

private:
    int w_ = 0, h_ = 0;
    std::vector<T> d_;
};

}  // namespace img
}  // namespace nvmm
