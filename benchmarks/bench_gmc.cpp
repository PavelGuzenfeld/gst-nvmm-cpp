/// bench_gmc — GMC (camera-motion) estimator micro-benchmark: compute us/frame and
/// shift error vs known shifts for each backend, on the same synthetic broadband
/// patches. Isolates the estimator from the pipeline (no VIC/engines), so the
/// numbers compare the backends' compute directly. Also doubles as a numeric gate:
/// exits non-zero if any backend mis-recovers a known shift.
///
///   ncc      — CPU zero-mean NCC brute-force (samurai_gmc.hpp)
///   fft-cpu  — CPU FFT phase correlation (gst/common/phase_correlation.hpp)
///   fft-cuda — VPI FFT (CUDA) phase correlation (gmc_vpi_fft.hpp)  [NVMM_HAVE_VPI]
///   pva      — VPI Harris+PyrLK (PVA) (gmc_vpi_pva.hpp)            [NVMM_HAVE_VPI]
///
/// CSV to stdout (two sections): timing (matches bench_nvmm convention) + accuracy.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "phase_correlation.hpp"
#include "samurai_gmc.hpp"
#ifdef NVMM_HAVE_VPI
#include "gmc_vpi_fft.hpp"
#include "gmc_vpi_pva.hpp"
#endif

using Clock = std::chrono::high_resolution_clock;
using Us = std::chrono::duration<double, std::micro>;

namespace {
constexpr int kIters = 300;

// Broadband noise base, big enough to crop shifted patches of any tested size.
std::vector<uint8_t> g_base;
int g_bw = 0, g_bh = 0;
uint8_t base_at(int x, int y) {
    x = ((x % g_bw) + g_bw) % g_bw; y = ((y % g_bh) + g_bh) % g_bh;
    return g_base[(size_t)y * g_bw + x];
}
void make_base(int w, int h) {
    g_bw = w; g_bh = h; g_base.resize((size_t)w * h);
    uint32_t s = 0x9e3779b9u;
    for (auto &v : g_base) { s = s * 1664525u + 1013904223u; v = (uint8_t)((s >> 24) & 0xFF); }
}
// prev/curr patches (n x n) where curr content is moved by (sx,sy) vs prev.
void make_pair(int n, int sx, int sy, std::vector<uint8_t> &prev, std::vector<uint8_t> &curr) {
    prev.resize((size_t)n * n); curr.resize((size_t)n * n);
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            prev[(size_t)y * n + x] = base_at(x, y);
            curr[(size_t)y * n + x] = base_at(x - sx, y - sy);
        }
}

struct Acc { double mean_err = 0, max_err = 0, mean_resp = 0; int nfail = 0; };

// Run `fn(prev,curr)->(dx,dy,resp)` over a shift set; return accuracy + gate.
template <typename Fn>
Acc accuracy(int n, double tol, Fn fn) {
    const int shifts[][2] = {{0, 0}, {2, 0}, {0, 3}, {3, -2}, {-4, 5}, {6, -3}, {-7, -5}};
    Acc a; int cnt = 0;
    std::vector<uint8_t> p, c;
    for (auto &sh : shifts) {
        make_pair(n, sh[0], sh[1], p, c);
        double dx, dy, resp; fn(p.data(), c.data(), dx, dy, resp);
        const double ex = std::fabs(dx - sh[0]), ey = std::fabs(dy - sh[1]);
        const double e = ex > ey ? ex : ey;
        a.mean_err += e; a.max_err = e > a.max_err ? e : a.max_err; a.mean_resp += resp;
        if (e > tol) a.nfail++;
        cnt++;
    }
    a.mean_err /= cnt; a.mean_resp /= cnt;
    return a;
}

// Median us/frame over kIters on a single representative pair.
template <typename Fn>
double timing(int n, Fn fn) {
    std::vector<uint8_t> p, c; make_pair(n, 5, -4, p, c);
    double dx, dy, resp;
    fn(p.data(), c.data(), dx, dy, resp);  // warm up
    double best = 1e18;
    for (int i = 0; i < kIters; i++) {
        const auto t0 = Clock::now();
        fn(p.data(), c.data(), dx, dy, resp);
        const double us = Us(Clock::now() - t0).count();
        if (us < best) best = us;
    }
    return best;  // min = least-noisy estimate of compute cost
}
}  // namespace

int main() {
    make_base(320, 320);
    int fails = 0;

    std::printf("benchmark,patch,min_us_per_frame\n");
    std::vector<std::pair<const char *, Acc>> accs;

    // --- ncc (128) ---
    {
        auto fn = [](const uint8_t *p, const uint8_t *c, double &dx, double &dy, double &r) {
            const nvmm::GmcShift s = nvmm::estimate_shift(p, c, 128, 24);
            dx = s.dx; dy = s.dy; r = s.conf;
        };
        std::printf("ncc,128,%.2f\n", timing(128, fn));
        accs.emplace_back("ncc", accuracy(128, 0.5, fn));
    }
    // --- fft-cpu (128) ---
    {
        nvmm::PhaseCorrelator pc(128, 128);
        std::vector<float> pf, cf;
        auto fn = [&](const uint8_t *p, const uint8_t *c, double &dx, double &dy, double &r) {
            pf.resize(128 * 128); cf.resize(128 * 128);
            for (int i = 0; i < 128 * 128; i++) { pf[i] = p[i]; cf[i] = c[i]; }
            const nvmm::PhaseCorrelator::Shift s = pc.correlate(pf.data(), cf.data());
            dx = s.x; dy = s.y; r = s.response;
        };
        std::printf("fft-cpu,128,%.2f\n", timing(128, fn));
        accs.emplace_back("fft-cpu", accuracy(128, 0.15, fn));
    }
#ifdef NVMM_HAVE_VPI
    // --- fft-cuda (128) ---
    if (nvmm::GmcVpiFft::available()) {
        nvmm::GmcVpiFft fft; std::string e;
        if (fft.init(128, e)) {
            auto fn = [&](const uint8_t *p, const uint8_t *c, double &dx, double &dy, double &r) {
                const nvmm::PhaseShift s = fft.estimate(p, c);
                dx = s.x; dy = s.y; r = s.response;
            };
            std::printf("fft-cuda,128,%.2f\n", timing(128, fn));
            accs.emplace_back("fft-cuda", accuracy(128, 0.15, fn));
        } else std::fprintf(stderr, "fft-cuda init failed: %s\n", e.c_str());
    } else std::fprintf(stderr, "fft-cuda unavailable on this box\n");

    // --- pva (256; Harris needs >=160) ---
    if (nvmm::GmcVpiPva::available()) {
        nvmm::GmcVpiPva pva; std::string e;
        if (pva.init(256, e)) {
            auto fn = [&](const uint8_t *p, const uint8_t *c, double &dx, double &dy, double &r) {
                const nvmm::GmcShift s = pva.estimate(p, c);
                dx = s.dx; dy = s.dy; r = s.conf;
            };
            std::printf("pva,256,%.2f\n", timing(256, fn));
            accs.emplace_back("pva", accuracy(256, 1.0, fn));  // feature-median: coarser
        } else std::fprintf(stderr, "pva init failed: %s\n", e.c_str());
    } else std::fprintf(stderr, "pva unavailable on this box\n");
#endif

    std::printf("\nbackend,mean_err_px,max_err_px,mean_resp,gate\n");
    for (auto &a : accs) {
        std::printf("%s,%.4f,%.4f,%.4f,%s\n", a.first, a.second.mean_err, a.second.max_err,
                    a.second.mean_resp, a.second.nfail ? "FAIL" : "ok");
        fails += a.second.nfail;
    }
    return fails ? 1 : 0;
}
