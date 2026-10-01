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

    template <typename U = T,
              typename std::enable_if<std::is_same<U, T>::value &&
                                      !std::is_const<U>::value, int>::type = 0>
    operator View<const U>() const { return View<const U>(data, width, height, stride); }

    View sub(const Rect &r) const {
        return View(data + (std::ptrdiff_t)r.y * stride + r.x, r.w, r.h, stride);
    }
};

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

}
}
