/// phase_correlation.hpp — sub-pixel translation estimation by FFT phase
/// correlation (host, dependency-free, unit-testable). OpenCV-free reimplementation
/// of cv::phaseCorrelate + cv::createHanningWindow, used by the SAMURAI GMC
/// (camera-motion compensation) path.
///
/// Method (matches cv::phaseCorrelate exactly so it is parity-checkable against it):
///   1. window both frames with a separable Hann window;
///   2. R = FFT(a) · conj(FFT(b)); C = R / |R|  (normalized cross-power spectrum);
///   3. c = real(IFFT(C)); fft-shift so DC lands at the centre;
///   4. locate the peak, refine to sub-pixel by a 5x5 weighted centroid;
///   5. shift = centre − peak;  response = centroid mass / (W·H)  in [0,1].
///
/// Dimensions MUST be powers of two (radix-2 Cooley–Tukey). cv::phaseCorrelate pads
/// to getOptimalDFTSize; feeding it pow2 dims makes both use the identical grid, so
/// the parity test is exact rather than approximate.
///
/// Pure C++ (<complex>), header-only, no OpenCV / CUDA / GStreamer.
#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace nvmm {

struct PhaseShift { double x = 0.0, y = 0.0, response = 0.0; };

// Refine a real correlation surface (w*h row-major, NOT yet fft-shifted) into a
// sub-pixel translation: fft-shift so DC lands at the centre, locate the peak,
// then a 5x5 weighted centroid (cv::weightedCentroid). Shared by PhaseCorrelator
// (CPU) and the VPI/CUDA FFT GMC backend so both use identical refinement. `surf`
// is shifted in place. Even dims only (GMC uses pow2). shift = centre − sub-pixel
// peak; response = centroid mass / (w·h).
inline PhaseShift refine_correlation_peak(std::vector<double> &surf, int w, int h) {
    const int hw = w / 2, hh = h / 2;
    for (int y = 0; y < hh; y++)
        for (int x = 0; x < w; x++) {
            const int xs = (x + hw) % w, ys = y + hh;
            std::swap(surf[(size_t)y * w + x], surf[(size_t)ys * w + xs]);
        }
    int px = 0, py = 0; double best = -1e300;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const double v = surf[(size_t)y * w + x];
            if (v > best) { best = v; px = x; py = y; }
        }
    double cx = 0.0, cy = 0.0, sum = 0.0;
    for (int y = std::max(0, py - 2); y <= std::min(h - 1, py + 2); y++)
        for (int x = std::max(0, px - 2); x <= std::min(w - 1, px + 2); x++) {
            const double v = surf[(size_t)y * w + x];
            cx += x * v; cy += y * v; sum += v;
        }
    PhaseShift out;
    out.response = sum / ((double)w * h);
    sum += 1e-15;
    cx /= sum; cy /= sum;
    out.x = (double)w / 2.0 - cx;
    out.y = (double)h / 2.0 - cy;
    return out;
}

class PhaseCorrelator {
public:
    struct Shift { double x = 0.0, y = 0.0, response = 0.0; };

    // w, h must be powers of two (radix-2 FFT); a non-pow2 dim would produce
    // silently wrong shifts, not a crash — so refuse it up front.
    PhaseCorrelator(int w, int h) : w_(w), h_(h) {
        assert(w_ > 1 && h_ > 1 && (w_ & (w_ - 1)) == 0 && (h_ & (h_ - 1)) == 0 &&
               "PhaseCorrelator dimensions must be powers of two");
        // Separable Hann window, matching cv::createHanningWindow:
        //   wc[i] = 0.5*(1 - cos(2*pi*i/(N-1))),  hann(y,x) = wr[y]*wc[x].
        wx_.resize((size_t)w_);
        wy_.resize((size_t)h_);
        for (int x = 0; x < w_; x++)
            wx_[(size_t)x] = 0.5 * (1.0 - std::cos(2.0 * kPi * x / (w_ - 1)));
        for (int y = 0; y < h_; y++)
            wy_[(size_t)y] = 0.5 * (1.0 - std::cos(2.0 * kPi * y / (h_ - 1)));
        A_.resize((size_t)w_ * h_);
        B_.resize((size_t)w_ * h_);
        col_.resize((size_t)std::max(w_, h_));
    }

    int width()  const { return w_; }
    int height() const { return h_; }

    // Sub-pixel shift such that a(x) ~= b(x + shift): the content moved by +shift
    // from a to b (same convention & sign as cv::phaseCorrelate(a, b, hann)).
    // `prev`/`curr` are row-major float, w*h each.
    Shift correlate(const float *prev, const float *curr) {
        for (int y = 0; y < h_; y++)
            for (int x = 0; x < w_; x++) {
                const double win = wy_[(size_t)y] * wx_[(size_t)x];
                const size_t i = (size_t)y * w_ + x;
                A_[i] = cd(prev[i] * win, 0.0);
                B_[i] = cd(curr[i] * win, 0.0);
            }
        fft2d(A_, false);
        fft2d(B_, false);
        // normalized cross-power spectrum, in place into A_
        for (size_t i = 0; i < A_.size(); i++) {
            const cd r = A_[i] * std::conj(B_[i]);
            const double m = std::abs(r);
            A_[i] = m > 1e-12 ? r / m : cd(0.0, 0.0);
        }
        fft2d(A_, true);   // unscaled inverse (matches cv::idft without DFT_SCALE)
        // Hand the real correlation surface to the shared refiner — the VPI/CUDA
        // FFT GMC backend fetches its IFFT result the same way, so both paths use
        // identical fft-shift + peak + 5x5-centroid refinement.
        real_.resize(A_.size());
        for (size_t i = 0; i < A_.size(); i++) real_[i] = A_[i].real();
        const PhaseShift ps = refine_correlation_peak(real_, w_, h_);
        return Shift{ps.x, ps.y, ps.response};
    }

private:
    using cd = std::complex<double>;
    static constexpr double kPi = 3.14159265358979323846;

    // In-place radix-2 Cooley–Tukey; NOT scaled on inverse (caller matches cv idft).
    static void fft1d(cd *a, int n, bool inverse) {
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(a[i], a[j]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            const double ang = 2.0 * kPi / len * (inverse ? 1.0 : -1.0);
            const cd wlen(std::cos(ang), std::sin(ang));
            for (int i = 0; i < n; i += len) {
                cd w(1.0, 0.0);
                for (int k = 0; k < len / 2; k++) {
                    const cd u = a[i + k], v = a[i + k + len / 2] * w;
                    a[i + k] = u + v;
                    a[i + k + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
    }

    void fft2d(std::vector<cd> &a, bool inverse) {
        for (int y = 0; y < h_; y++) fft1d(&a[(size_t)y * w_], w_, inverse);
        for (int x = 0; x < w_; x++) {
            for (int y = 0; y < h_; y++) col_[(size_t)y] = a[(size_t)y * w_ + x];
            fft1d(col_.data(), h_, inverse);
            for (int y = 0; y < h_; y++) a[(size_t)y * w_ + x] = col_[(size_t)y];
        }
    }

    int w_, h_;
    std::vector<double> wx_, wy_;
    std::vector<cd> A_, B_, col_;
    std::vector<double> real_;  // scratch: real part of the IFFT surface for refine
};

}  // namespace nvmm
