#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace nvmm {

struct GmcShift { float dx = 0.f, dy = 0.f, conf = 0.f; };

/// Integer-px shift in patch units with curr[y,x] ~= prev[y-dy, x-dx]. Zero-mean
/// NCC; conf is the peak correlation in [-1,1].
inline GmcShift estimate_shift(const uint8_t *prev, const uint8_t *curr,
                               int n, int search)
{
    double mp = 0, mc = 0;
    for (int i = 0; i < n * n; i++) { mp += prev[i]; mc += curr[i]; }
    mp /= (double)n * n; mc /= (double)n * n;

    GmcShift best; double best_ncc = -2.0;
    for (int sy = -search; sy <= search; sy++) {
        const int y0 = sy > 0 ? sy : 0, y1 = n + (sy < 0 ? sy : 0);
        for (int sx = -search; sx <= search; sx++) {
            const int x0 = sx > 0 ? sx : 0, x1 = n + (sx < 0 ? sx : 0);
            double s = 0, sa = 0, sb = 0;
            for (int y = y0; y < y1; y++) {
                const uint8_t *cr = curr + (size_t)y * n;
                const uint8_t *pr = prev + (size_t)(y - sy) * n;
                for (int x = x0; x < x1; x++) {
                    const double a = cr[x] - mc, b = pr[x - sx] - mp;
                    s += a * b; sa += a * a; sb += b * b;
                }
            }
            if (sa > 0 && sb > 0) {
                const double ncc = s / std::sqrt(sa * sb);
                if (ncc > best_ncc) { best_ncc = ncc; best.dx = (float)sx; best.dy = (float)sy; }
            }
        }
    }
    best.conf = (float)best_ncc;
    return best;
}

}
