/// seed_math_probe — validate samurai_seed_math (bilinear 128->512 + binarize)
/// against the Phase-A golden: the captured seed-frame decoder mask vs the mask
/// the Python memory_encoder actually received (in_mask). Colorspace-free, pure
/// host (no CUDA/TRT/GST).
///
/// Usage: seed_math_probe <dec_masks.bin 1x1x128x128> <in_mask.bin 1x1x512x512>
#include <cmath>
#include <cstdio>
#include <vector>

#include "samurai_seed_math.hpp"

static std::vector<float> readf(const char *p, size_t n)
{
    std::vector<float> v(n);
    FILE *f = std::fopen(p, "rb");
    if (!f || std::fread(v.data(), sizeof(float), n, f) != n) { std::fprintf(stderr, "read %s failed\n", p); v.clear(); }
    if (f) std::fclose(f);
    return v;
}

int main(int argc, char **argv)
{
    if (argc != 3) { std::fprintf(stderr, "usage: %s dec_masks.bin in_mask.bin\n", argv[0]); return 2; }
    const int M = 128, HI = 512;
    auto low = readf(argv[1], (size_t)M * M);
    auto gold = readf(argv[2], (size_t)HI * HI);
    if (low.empty() || gold.empty()) return 1;

    std::vector<float> high = nvmm::bilinear_upsample(low.data(), M, M, HI, HI);
    std::vector<float> mem((size_t)HI * HI);
    int mism = 0;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < mem.size(); i++) {
        mem[i] = (high[i] > 0.f) ? 10.f : -10.f;
        if (mem[i] != gold[i]) mism++;
        dot += (double)mem[i] * gold[i]; na += (double)mem[i] * mem[i]; nb += (double)gold[i] * gold[i];
    }
    nvmm::MaskBox b = nvmm::mask_to_box(high.data(), HI, HI, 0.f);
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    int gpos = 0, mpos = 0;
    for (size_t i = 0; i < gold.size(); i++) { if (gold[i] > 0) gpos++; if (mem[i] > 0) mpos++; }
    std::printf("mem vs golden in_mask: cos=%.6f mismatched_px=%d/%zu (mine>0=%d gold>0=%d)\n",
                cos, mism, mem.size(), mpos, gpos);
    std::printf("mask_to_box: valid=%d box=(%.0f,%.0f %.0fx%.0f)\n", b.valid, b.x, b.y, b.w, b.h);
    std::printf("%s\n", (cos >= 0.999 && mism <= 8) ? "SEED_MASK_PARITY_PASS" : "SEED_MASK_PARITY_CHECK");
    return 0;
}
