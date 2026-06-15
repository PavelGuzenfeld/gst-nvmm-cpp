/// samurai_mem_probe — validate samurai_memory.hpp assemble_memory() against the
/// frame-60 golden (memattn_real). Anchorable slices (cond frame never changes,
/// obj_pos depends only on pos_list): memory[0:1024] (cond maskmem content),
/// memory_pos[0:1024] (cond pos + tpos[6]), memory[7168:7172] (cond obj_ptr split),
/// memory_pos[7168:7232] (obj_pos for pos_list=[60,1..15]). Non-cond slots are
/// filled with the cond frame (cold-start) and not checked. Pure host.
///
/// Usage: samurai_mem_probe <parity_dir>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "samurai_memory.hpp"

using namespace nvmm::memdims;

static std::vector<float> rd(const std::string &p, size_t n)
{
    std::vector<float> v(n);
    FILE *f = std::fopen(p.c_str(), "rb");
    if (!f || std::fread(v.data(), sizeof(float), n, f) != n) { std::fprintf(stderr, "read %s\n", p.c_str()); v.clear(); }
    if (f) std::fclose(f);
    return v;
}
static double cosine(const float *a, const float *b, size_t n)
{
    double d = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return d / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}
static double maxabs(const float *a, const float *b, size_t n)
{
    double m = 0;
    for (size_t i = 0; i < n; i++) m = std::fmax(m, std::fabs((double)a[i] - b[i]));
    return m;
}

int main(int argc, char **argv)
{
    if (argc != 2) { std::fprintf(stderr, "usage: %s parity_dir\n", argv[0]); return 2; }
    const std::string P = std::string(argv[1]) + "/";
    auto feat = rd(P + "mem_cond_feat.bin", (size_t)kMem * kTok);
    auto cpos = rd(P + "mem_cond_pos.bin", (size_t)kMem * kTok);
    auto tpos = rd(P + "mem_tpos.bin", (size_t)kMask * kMem);
    auto pw = rd(P + "mem_tposproj_w.bin", (size_t)kMem * kHid);
    auto pb = rd(P + "mem_tposproj_b.bin", (size_t)kMem);
    auto ptr = rd(P + "mem_cond_ptr.bin", (size_t)kHid);
    auto gmem = rd(P + "golden_memory.bin", (size_t)kTotal * kMem);
    auto gpos = rd(P + "golden_mempos.bin", (size_t)kTotal * kMem);
    if (feat.empty() || cpos.empty() || tpos.empty() || pw.empty() || pb.empty() ||
        ptr.empty() || gmem.empty() || gpos.empty()) return 1;

    // Cold-start fill: every maskmem slot + obj_ptr = the cond frame.
    const float *maskmem[kMask]; for (int s = 0; s < kMask; s++) maskmem[s] = feat.data();
    const float *objptr[kPtr];  for (int p = 0; p < kPtr; p++) objptr[p] = ptr.data();
    float pos_list[kPtr]; pos_list[0] = 60.f;                  // cond = frame_idx
    for (int p = 1; p < kPtr; p++) pos_list[p] = (float)p;     // t_diff 1..15

    std::vector<float> memory((size_t)kTotal * kMem), mempos((size_t)kTotal * kMem);
    nvmm::MemConsts c{cpos.data(), tpos.data(), pw.data(), pb.data()};
    nvmm::assemble_memory(maskmem, objptr, pos_list, c, memory.data(), mempos.data());

    auto row = [](std::vector<float> &v, int r) { return v.data() + (size_t)r * kMem; };
    // cond maskmem content (cosine — fp16 noise between capture runs).
    double c0 = cosine(row(memory, 0), row(gmem, 0), (size_t)kTok * kMem);
    // cond maskmem pos = cond_pos + tpos[6] (tight).
    double p0 = maxabs(row(mempos, 0), row(gpos, 0), (size_t)kTok * kMem);
    // cond obj_ptr split (rows 7168..7171).
    double cp = maxabs(row(memory, kMask * kTok), row(gmem, kMask * kTok), (size_t)kPtrTok * kMem);
    // obj_pos block (rows 7168..7231) — depends only on pos_list.
    double op = maxabs(row(mempos, kMask * kTok), row(gpos, kMask * kTok), (size_t)kObjTok * kMem);

    std::printf("cond maskmem content : cosine=%.6f %s\n", c0, c0 >= 0.999 ? "PASS" : "FAIL");
    std::printf("cond maskmem pos+tpos: maxabs=%.4g %s\n", p0, p0 < 1e-2 ? "PASS" : "FAIL");
    std::printf("cond obj_ptr split   : maxabs=%.4g %s\n", cp, cp < 1e-2 ? "PASS" : "FAIL");
    std::printf("obj_pos temporal PE  : maxabs=%.4g %s\n", op, op < 1e-2 ? "PASS" : "FAIL");
    bool ok = c0 >= 0.999 && p0 < 1e-2 && cp < 1e-2 && op < 1e-2;
    std::printf("%s\n", ok ? "MEM_ASSEMBLE_PASS" : "MEM_ASSEMBLE_FAIL");
    return ok ? 0 : 1;
}
