#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "samurai_kernels.hpp"
#include "samurai_seed_math.hpp"
#include "samurai_memory.hpp"
#include "samurai_consts.hpp"

static float pat(int i) { return std::sin(0.1f * i) * 3.f - 1.f; }
static float *dev(const std::vector<float> &h)
{
    float *d = nullptr; cudaMalloc(&d, h.size() * sizeof(float));
    cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}
static std::vector<float> host(const float *d, size_t n)
{
    std::vector<float> h(n); cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}
static double maxabs(const std::vector<float> &a, const std::vector<float> &b)
{
    double m = 0; for (size_t i = 0; i < a.size(); i++) m = std::fmax(m, std::fabs((double)a[i] - b[i]));
    return m;
}

/// The last element ends where a PROT_NONE page begins, so a read or write one
/// element past the end faults instead of touching whatever the heap holds.
class GuardedFloats {
public:
    explicit GuardedFloats(const std::vector<float> &v) : n_(v.size())
    {
        const size_t page = (size_t)sysconf(_SC_PAGESIZE);
        bytes_ = (n_ * sizeof(float) + page - 1) / page * page + page;
        base_ = static_cast<char *>(mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (base_ == MAP_FAILED || mprotect(base_ + bytes_ - page, page, PROT_NONE) != 0)
            std::abort();
        data_ = reinterpret_cast<float *>(base_ + bytes_ - page) - n_;
        std::copy(v.begin(), v.end(), data_);
    }
    GuardedFloats(const GuardedFloats &) = delete;
    GuardedFloats &operator=(const GuardedFloats &) = delete;
    ~GuardedFloats() { munmap(base_, bytes_); }
    float *data() { return data_; }
    std::vector<float> copy() const { return std::vector<float>(data_, data_ + n_); }

private:
    size_t n_, bytes_ = 0;
    char *base_ = nullptr;
    float *data_ = nullptr;
};

constexpr uint32_t kConstsMagic = 0x5341434Eu;
constexpr float kSentinel = -7777.f;

/// __expf is within 2 + floor(1.173|x|) ulp (CUDA Programming Guide, intrinsic functions):
/// 6 ulp at |x| <= 4, under 7.2e-7 relative; sigmoid' <= 1/4 and the 20x scale make that
/// 3.6e-6, plus four roundings at ulp(10).
constexpr double kSigmoidBound = 1e-5;
/// nvcc fuses each of the three lerps into an fma; with |b - a| < 8 each fused product
/// skips one rounding of at most 2^-22, so the result moves by a few of those.
constexpr double kBilinearBound = 1e-6;
/// Two 256-term float sums of |terms| <= 14.3 differ by <= 2*gamma_255*14.3 = 4.4e-4
/// (Higham); __sinf/__cosf add <= 2^-21.19 per tap on |x| <= pi (cond pos 45 keeps
/// x <= 3), times sum|w| <= 10.3.
constexpr double kAssemblePosBound = 5e-4;

struct ConstsFile {
    std::vector<unsigned char> bytes;
    ConstsFile &u32(uint32_t v)
    {
        for (int b = 0; b < 4; b++) bytes.push_back((unsigned char)(v >> (8 * b)));
        return *this;
    }
    ConstsFile &text(const std::string &s)
    {
        bytes.insert(bytes.end(), s.begin(), s.end());
        return *this;
    }
    ConstsFile &f32(float v)
    {
        unsigned char b[4];
        std::memcpy(b, &v, 4);
        bytes.insert(bytes.end(), b, b + 4);
        return *this;
    }
    ConstsFile &tensor(const std::string &name, const std::vector<uint32_t> &dims,
                       const std::vector<float> &data)
    {
        u32((uint32_t)name.size()).text(name).u32((uint32_t)dims.size());
        for (uint32_t d : dims) u32(d);
        for (float v : data) f32(v);
        return *this;
    }
    ConstsFile truncated(size_t drop) const
    {
        ConstsFile t;
        t.bytes.assign(bytes.begin(), bytes.end() - (std::ptrdiff_t)drop);
        return t;
    }
};

static bool load_consts(const ConstsFile &file, nvmm::SamuraiConsts &c, std::string &err)
{
    char path[] = "/tmp/samurai_consts_probe_XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) std::abort();
    const ssize_t written = write(fd, file.bytes.data(), file.bytes.size());
    close(fd);
    if (written != (ssize_t)file.bytes.size()) std::abort();
    const bool ok = c.load(path, err);
    unlink(path);
    return ok;
}

static bool load_fails_with(const ConstsFile &file, const std::string &expected)
{
    nvmm::SamuraiConsts c;
    std::string err;
    const bool ok = load_consts(file, c, err);
    if (ok || err != expected) std::printf("    got ok=%d err=\"%s\"\n", ok, err.c_str());
    return !ok && err == expected;
}

static ConstsFile one_tensor_file()
{
    return ConstsFile().u32(kConstsMagic).u32(1).tensor("pos", {4}, {1.f, 2.f, 3.f, 4.f});
}

static bool consts_round_trip_keeps_names_shapes_and_values()
{
    const std::vector<float> grid = {0.5f, -1.f, 2.25f, 3.f, -4.5f, 5.f, 6.f, 7.f,
                                     8.f, 9.f, 10.f, 11.f, 12.f, 13.f, -14.f};
    const ConstsFile file = ConstsFile().u32(kConstsMagic).u32(3)
        .tensor("grid", {3, 5}, grid)
        .tensor("bias", {2}, {-0.25f, 0.75f})
        .tensor("scale", {}, {42.f});
    nvmm::SamuraiConsts c;
    std::string err;
    if (!load_consts(file, c, err)) { std::printf("    err=\"%s\"\n", err.c_str()); return false; }
    const nvmm::ConstTensor *g = c.get("grid"), *b = c.get("bias"), *s = c.get("scale");
    return c.size() == 3 && g && b && s &&
           g->shape == std::vector<int>{3, 5} && g->count() == 15 && g->data == grid &&
           b->shape == std::vector<int>{2} && b->data == std::vector<float>{-0.25f, 0.75f} &&
           s->shape.empty() && s->data == std::vector<float>{42.f} &&
           c.data("bias") == b->data.data() && c.data("bias")[1] == 0.75f;
}

static bool consts_unknown_name_is_null()
{
    nvmm::SamuraiConsts c;
    std::string err;
    return load_consts(one_tensor_file(), c, err) && c.get("po") == nullptr &&
           c.data("missing") == nullptr && c.get("pos") != nullptr;
}

static bool consts_missing_file_is_reported()
{
    nvmm::SamuraiConsts c;
    std::string err;
    return !c.load("/nonexistent/samurai_consts.bin", err) &&
           err == "cannot open consts: /nonexistent/samurai_consts.bin" && c.size() == 0;
}

static bool consts_rejects_wrong_magic()
{
    return load_fails_with(ConstsFile().u32(kConstsMagic + 1).u32(0), "bad consts magic/header");
}

static bool consts_rejects_header_shorter_than_count()
{
    return load_fails_with(ConstsFile().u32(kConstsMagic).u32(0).truncated(1),
                           "bad consts magic/header") &&
           load_fails_with(ConstsFile().u32(kConstsMagic).truncated(2), "bad consts magic/header");
}

static bool consts_rejects_count_beyond_tensors()
{
    ConstsFile f = one_tensor_file();
    f.bytes[4] = 2;
    return load_fails_with(f, "truncated name len");
}

static bool consts_rejects_short_name()
{
    return load_fails_with(ConstsFile().u32(kConstsMagic).u32(1).u32(5).text("pos"),
                           "truncated name");
}

static bool consts_rejects_missing_ndim()
{
    return load_fails_with(ConstsFile().u32(kConstsMagic).u32(1).u32(3).text("pos"),
                           "truncated ndim");
}

static bool consts_rejects_missing_dim()
{
    return load_fails_with(ConstsFile().u32(kConstsMagic).u32(1).u32(3).text("pos").u32(2).u32(4),
                           "truncated dim");
}

static bool consts_rejects_data_shorter_than_shape()
{
    return load_fails_with(one_tensor_file().truncated(4), "truncated data for pos") &&
           load_fails_with(one_tensor_file().truncated(1), "truncated data for pos");
}

static bool memdims_match_sam2_512_crop()
{
    using namespace nvmm::memdims;
    return kTok == 1024 && kMem == 64 && kHid == 256 && kMask == 7 && kPtr == 16 &&
           kPtrTok == 4 && kObjTok == 64 && kTotal == 7232 && kTdiffMax == 15.f;
}

static bool mask_box_reaches_all_four_edges()
{
    const int h = 5, w = 6;
    std::vector<float> m((size_t)h * w, 0.f);
    m[0 * w + 3] = 1.f; m[4 * w + 1] = 1.f; m[2 * w + 0] = 1.f; m[3 * w + 5] = 1.f;
    GuardedFloats g(m);
    const nvmm::MaskBox b = nvmm::mask_to_box(g.data(), h, w, 0.f);
    return b.valid && b.x == 0.f && b.y == 0.f && b.w == 5.f && b.h == 4.f;
}

static bool mask_box_excludes_pixels_equal_to_thresh()
{
    const int h = 4, w = 5;
    std::vector<float> m((size_t)h * w, 0.5f);
    m[2 * w + 3] = 0.75f;
    GuardedFloats g(m);
    const nvmm::MaskBox b = nvmm::mask_to_box(g.data(), h, w, 0.5f);
    return b.valid && b.x == 3.f && b.y == 2.f && b.w == 0.f && b.h == 0.f;
}

static bool mask_box_single_pixel_in_column_zero_is_valid()
{
    const int h = 3, w = 4;
    std::vector<float> m((size_t)h * w, -1.f);
    m[1 * w + 0] = 2.f;
    GuardedFloats g(m);
    const nvmm::MaskBox b = nvmm::mask_to_box(g.data(), h, w, 0.f);
    return b.valid && b.x == 0.f && b.y == 1.f && b.w == 0.f && b.h == 0.f;
}

static bool mask_box_of_empty_mask_is_invalid()
{
    std::vector<float> m(12, 0.f);
    GuardedFloats g(m);
    const nvmm::MaskBox b = nvmm::mask_to_box(g.data(), 3, 4, 0.f);
    return !b.valid && b.x == 0.f && b.y == 0.f && b.w == 0.f && b.h == 0.f;
}

/// Integer weights keep every product and sum exact, so the expected values are exact.
static bool mlp3_relu_clamps_hidden_layers_only()
{
    GuardedFloats x({1.f, -2.f});
    GuardedFloats w0({1.f, 2.f, 3.f, -1.f}), b0({0.5f, -1.f});
    GuardedFloats w1({2.f, 1.f, 1.f, -1.f}), b1({-3.f, 1.f});
    GuardedFloats w2({-2.f, 5.f, 1.f, 1.f}), b2({0.5f, -3.f});
    const std::vector<float> out = nvmm::mlp3_relu(x.data(), 2, w0.data(), b0.data(),
                                                   w1.data(), b1.data(), w2.data(), b2.data());
    return out == std::vector<float>{-1.5f, -2.f};
}

static bool written_once_each(const std::vector<float> &out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        if (out[i] == kSentinel) return false;
    return out[n] == kSentinel;
}

static float *dev_sentinel(size_t n) { return dev(std::vector<float>(n, kSentinel)); }

constexpr int kRows = 27, kCols = 19, kN = kRows * kCols;

static bool transpose_covers_one_past_two_blocks(cudaStream_t s)
{
    std::vector<float> in(kN);
    for (int i = 0; i < kN; i++) in[i] = (float)i;
    float *di = dev(in), *dout = dev_sentinel(kN + 1);
    nvmm::k_transpose(di, dout, kRows, kCols, s); cudaStreamSynchronize(s);
    std::vector<float> got = host(dout, kN + 1), ref(kN + 1, kSentinel);
    for (int r = 0; r < kRows; r++) for (int c = 0; c < kCols; c++) ref[c * kRows + r] = in[r * kCols + c];
    cudaFree(di); cudaFree(dout);
    return got == ref;
}

static bool add_per_channel_covers_one_past_two_blocks(cudaStream_t s)
{
    std::vector<float> in(kN), bias(kRows);
    for (int i = 0; i < kN; i++) in[i] = pat(i);
    for (int c = 0; c < kRows; c++) bias[c] = (float)(c + 1);
    float *di = dev(in), *db = dev(bias), *dout = dev_sentinel(kN + 1);
    nvmm::k_add_per_channel(di, db, dout, kRows, kCols, s); cudaStreamSynchronize(s);
    std::vector<float> got = host(dout, kN + 1), ref(kN + 1, kSentinel);
    for (int i = 0; i < kN; i++) ref[i] = in[i] + bias[i / kCols];
    cudaFree(di); cudaFree(db); cudaFree(dout);
    return got == ref;
}

static bool sigmoid_writes_each_element_once(cudaStream_t s)
{
    std::vector<float> in(kN);
    for (int i = 0; i < kN; i++) in[i] = pat(i);
    float *di = dev(in), *dout = dev_sentinel(kN + 1);
    nvmm::k_sigmoid_scale(di, dout, kN, 20.f, -10.f, s); cudaStreamSynchronize(s);
    const bool ok = written_once_each(host(dout, kN + 1), kN);
    cudaFree(di); cudaFree(dout);
    return ok;
}

static bool threshold_zero_maps_to_lo(cudaStream_t s)
{
    std::vector<float> in(kN);
    for (int i = 0; i < kN; i++) in[i] = (i % 3 == 0) ? 0.f : pat(i);
    float *di = dev(in), *dout = dev_sentinel(kN + 1);
    nvmm::k_threshold_scale(di, dout, kN, 10.f, -10.f, s); cudaStreamSynchronize(s);
    std::vector<float> got = host(dout, kN + 1), ref(kN + 1, kSentinel);
    for (int i = 0; i < kN; i++) ref[i] = in[i] > 0.f ? 10.f : -10.f;
    cudaFree(di); cudaFree(dout);
    return got == ref;
}

static bool bilinear_identity_reproduces_one_past_a_block(cudaStream_t s)
{
    const int w = 257;
    std::vector<float> in(w);
    for (int i = 0; i < w; i++) in[i] = pat(i);
    float *di = dev(in), *dout = dev_sentinel(w + 1);
    nvmm::k_bilinear(di, dout, 1, w, 1, w, s); cudaStreamSynchronize(s);
    std::vector<float> got = host(dout, w + 1), ref = in;
    ref.push_back(kSentinel);
    cudaFree(di); cudaFree(dout);
    return got == ref && nvmm::bilinear_upsample(in.data(), 1, w, 1, w) == in;
}

static bool mask_bbox_ignores_zero_and_reaches_last_pixel(cudaStream_t s)
{
    std::vector<float> m(kN + 1, 0.f);
    m[3 * kCols + 4] = 1.f;
    m[kN - 1] = 1.f;
    m[kN] = 1.f;
    float *dm = dev(m); int *dbox; cudaMalloc(&dbox, 4 * sizeof(int));
    int box[4] = {kCols, kRows, -1, -1};
    cudaMemcpy(dbox, box, sizeof(box), cudaMemcpyHostToDevice);
    nvmm::k_mask_bbox(dm, kRows, kCols, dbox, s); cudaStreamSynchronize(s);
    cudaMemcpy(box, dbox, sizeof(box), cudaMemcpyDeviceToHost);
    cudaFree(dm); cudaFree(dbox);
    return box[0] == 4 && box[1] == 3 && box[2] == kCols - 1 && box[3] == kRows - 1;
}

/// tok = 9 (crop 48) leaves (7*9+64)*64 short of a block multiple, so threads past
/// the end exist and the tail sentinel sees any write they make.
static bool assemble_small_tok_keeps_layout_and_tail(cudaStream_t s)
{
    const int tok = 9, rows = 7 * tok + 64, n = rows * 64;
    std::vector<float> maskmem[7], objptr(16 * 256), pos(16), mpos((size_t)64 * tok),
        tpos(7 * 64), tpw(64 * 256, 0.f), tpb(64, 0.f);
    for (int sl = 0; sl < 7; sl++) {
        maskmem[sl].resize((size_t)64 * tok);
        for (size_t i = 0; i < maskmem[sl].size(); i++) maskmem[sl][i] = (float)(sl * 1000 + (int)i);
    }
    for (size_t i = 0; i < objptr.size(); i++) objptr[i] = (float)(-(int)i);
    for (int p = 0; p < 16; p++) pos[p] = (float)p;
    for (size_t i = 0; i < mpos.size(); i++) mpos[i] = (float)i;
    for (size_t i = 0; i < tpos.size(); i++) tpos[i] = 0.5f * (float)i;
    float *dmm[7]; for (int sl = 0; sl < 7; sl++) dmm[sl] = dev(maskmem[sl]);
    float *dptrs; cudaMalloc(&dptrs, sizeof(dmm)); cudaMemcpy(dptrs, dmm, sizeof(dmm), cudaMemcpyHostToDevice);
    float *dop = dev(objptr), *dpos = dev(pos), *dmp = dev(mpos), *dtp = dev(tpos),
          *dtw = dev(tpw), *dtb = dev(tpb);
    float *dmem = dev_sentinel(n + 64), *dmpos = dev_sentinel(n + 64);
    nvmm::k_assemble_memory(reinterpret_cast<const float *const *>(dptrs), dop, dpos, dmp, dtp,
                            dtw, dtb, dmem, dmpos, tok, s);
    cudaStreamSynchronize(s);
    const std::vector<float> mem = host(dmem, n + 64), mp = host(dmpos, n + 64);
    bool ok = true;
    for (int sl = 0; sl < 7; sl++)
        for (int i = 0; i < tok; i++)
            for (int ch = 0; ch < 64; ch++) {
                const size_t at = (size_t)(sl * tok + i) * 64 + ch;
                ok = ok && mem[at] == maskmem[sl][(size_t)ch * tok + i] &&
                     mp[at] == mpos[(size_t)ch * tok + i] + tpos[(6 - sl) * 64 + ch];
            }
    for (int p = 0; p < 16; p++)
        for (int k = 0; k < 4; k++)
            for (int ch = 0; ch < 64; ch++)
                ok = ok && mem[(size_t)(7 * tok + p * 4 + k) * 64 + ch] == objptr[p * 256 + k * 64 + ch];
    for (int i = n; i < n + 64; i++) ok = ok && mem[i] == kSentinel && mp[i] == kSentinel;
    for (int sl = 0; sl < 7; sl++) cudaFree(dmm[sl]);
    cudaFree(dptrs); cudaFree(dop); cudaFree(dpos); cudaFree(dmp); cudaFree(dtp); cudaFree(dtw);
    cudaFree(dtb); cudaFree(dmem); cudaFree(dmpos);
    return ok;
}

/// Hann taps are quarters and pixels integers, so every product is exact.
static bool gmc_window_reads_pitched_rows_exactly(cudaStream_t s)
{
    const int n = 17, pitch = 20;
    std::vector<unsigned char> y((size_t)pitch * n);
    for (size_t i = 0; i < y.size(); i++) y[i] = (unsigned char)((i * 7 + 3) % 251);
    std::vector<float> hann(n);
    for (int i = 0; i < n; i++) hann[i] = 0.25f * (float)(i % 4 + 1);
    unsigned char *dy; cudaMalloc(&dy, y.size());
    cudaMemcpy(dy, y.data(), y.size(), cudaMemcpyHostToDevice);
    float *dh = dev(hann);
    float *dout = dev_sentinel(2 * (n * n + 1));
    nvmm::k_gmc_window(dy, pitch, n, dh, reinterpret_cast<float2 *>(dout), s);
    cudaStreamSynchronize(s);
    std::vector<float> got = host(dout, 2 * (n * n + 1)), ref(2 * (n * n + 1), kSentinel);
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
            ref[2 * (r * n + c)] = (float)y[(size_t)r * pitch + c] * (hann[r] * hann[c]);
            ref[2 * (r * n + c) + 1] = 0.f;
        }
    cudaFree(dy); cudaFree(dh); cudaFree(dout);
    return got == ref;
}

/// R = a*conj(b) is a Pythagorean pair, so |R| and R/|R| are exact; the third case
/// puts |R| exactly on the 1e-12f cutoff, which must zero like PhaseCorrelator's m > eps.
static bool gmc_cross_power_normalises_and_zeroes_at_cutoff(cudaStream_t s)
{
    const int n2 = 257;
    const float a_cases[4][2] = {{2.f, 1.f}, {0.f, 0.f}, {1e-12f, 0.f}, {1.f, 2.f}};
    const float b_cases[4][2] = {{2.f, -1.f}, {1.f, 2.f}, {1.f, 0.f}, {2.f, 1.f}};
    const float r_cases[4][2] = {{3.f / 5.f, 4.f / 5.f}, {0.f, 0.f}, {0.f, 0.f}, {4.f / 5.f, 3.f / 5.f}};
    std::vector<float> a(2 * (n2 + 1), kSentinel), b(2 * (n2 + 1), kSentinel),
        ref(2 * (n2 + 1), kSentinel);
    for (int i = 0; i < n2; i++)
        for (int k = 0; k < 2; k++) {
            a[2 * i + k] = a_cases[i % 4][k];
            b[2 * i + k] = b_cases[i % 4][k];
            ref[2 * i + k] = r_cases[i % 4][k];
        }
    float *da = dev(a), *db = dev(b);
    nvmm::k_gmc_cross_power(reinterpret_cast<float2 *>(da), reinterpret_cast<const float2 *>(db),
                            n2, s);
    cudaStreamSynchronize(s);
    std::vector<float> got = host(da, a.size());
    cudaFree(da); cudaFree(db);
    return got == ref;
}

int main()
{
    cudaStream_t s; cudaStreamCreate(&s);
    bool ok = true;
    auto report = [&](const char *n, double e, double tol) {
        bool p = e <= tol; ok = ok && p;
        std::printf("  %-16s maxabs=%.4g %s\n", n, e, p ? "PASS" : "FAIL");
    };
    auto expect = [&](const char *n, bool p) {
        ok = ok && p;
        std::printf("  %-48s %s\n", n, p ? "PASS" : "FAIL");
    };

    { int C = 5, HW = 7; std::vector<float> in(C * HW); for (int i = 0; i < C * HW; i++) in[i] = pat(i);
      float *di = dev(in), *dout; cudaMalloc(&dout, in.size() * sizeof(float));
      nvmm::k_transpose(di, dout, C, HW, s); cudaStreamSynchronize(s);
      auto g = host(dout, in.size()); std::vector<float> ref(C * HW);
      for (int r = 0; r < C; r++) for (int c = 0; c < HW; c++) ref[c * C + r] = in[r * HW + c];
      report("transpose", maxabs(g, ref), 0); cudaFree(di); cudaFree(dout); }

    { int C = 4, HW = 6; std::vector<float> in(C * HW), bias(C);
      for (int i = 0; i < C * HW; i++) in[i] = pat(i);
      for (int c = 0; c < C; c++) bias[c] = pat(100 + c);
      float *di = dev(in), *db = dev(bias), *dout; cudaMalloc(&dout, in.size() * sizeof(float));
      nvmm::k_add_per_channel(di, db, dout, C, HW, s); cudaStreamSynchronize(s);
      auto g = host(dout, in.size()); std::vector<float> ref(C * HW);
      for (int c = 0; c < C; c++) for (int i = 0; i < HW; i++) ref[c * HW + i] = in[c * HW + i] + bias[c];
      report("add_per_channel", maxabs(g, ref), 0); cudaFree(di); cudaFree(db); cudaFree(dout); }

    { int n = 50; std::vector<float> in(n); for (int i = 0; i < n; i++) in[i] = pat(i);
      float *di = dev(in), *dout; cudaMalloc(&dout, n * sizeof(float));
      nvmm::k_sigmoid_scale(di, dout, n, 20.f, -10.f, s); cudaStreamSynchronize(s);
      auto g = host(dout, n); std::vector<float> ref(n);
      for (int i = 0; i < n; i++) ref[i] = (1.f / (1.f + std::exp(-in[i]))) * 20.f - 10.f;
      report("sigmoid_scale", maxabs(g, ref), kSigmoidBound); cudaFree(di); cudaFree(dout); }

    { int n = 50; std::vector<float> in(n); for (int i = 0; i < n; i++) in[i] = pat(i);
      float *di = dev(in), *dout; cudaMalloc(&dout, n * sizeof(float));
      nvmm::k_threshold_scale(di, dout, n, 10.f, -10.f, s); cudaStreamSynchronize(s);
      auto g = host(dout, n); std::vector<float> ref(n);
      for (int i = 0; i < n; i++) ref[i] = in[i] > 0.f ? 10.f : -10.f;
      report("threshold_scale", maxabs(g, ref), 0); cudaFree(di); cudaFree(dout); }

    { int hi = 8, wi = 8, ho = 16, wo = 16; std::vector<float> in(hi * wi);
      for (int i = 0; i < hi * wi; i++) in[i] = pat(i);
      float *di = dev(in), *dout; cudaMalloc(&dout, (size_t)ho * wo * sizeof(float));
      nvmm::k_bilinear(di, dout, hi, wi, ho, wo, s); cudaStreamSynchronize(s);
      auto g = host(dout, (size_t)ho * wo);
      auto ref = nvmm::bilinear_upsample(in.data(), hi, wi, ho, wo);
      report("bilinear", maxabs(g, ref), kBilinearBound); cudaFree(di); cudaFree(dout); }

    { int h = 20, w = 24; std::vector<float> m((size_t)h * w, -1.f);
      for (int y = 5; y <= 12; y++) for (int x = 7; x <= 17; x++) m[y * w + x] = 1.f;
      float *dm = dev(m); int *dbox; cudaMalloc(&dbox, 4 * sizeof(int));
      int init[4] = {w, h, -1, -1}; cudaMemcpy(dbox, init, sizeof(init), cudaMemcpyHostToDevice);
      nvmm::k_mask_bbox(dm, h, w, dbox, s); cudaStreamSynchronize(s);
      int box[4]; cudaMemcpy(box, dbox, sizeof(box), cudaMemcpyDeviceToHost);
      nvmm::MaskBox ref = nvmm::mask_to_box(m.data(), h, w, 0.f);
      double e = std::abs(box[0] - ref.x) + std::abs(box[1] - ref.y) +
                 std::abs((box[2] - box[0]) - ref.w) + std::abs((box[3] - box[1]) - ref.h);
      report("mask_bbox", e, 0); cudaFree(dm); cudaFree(dbox); }

    {
        using namespace nvmm::memdims;
        std::vector<float> maskmem[kMask], objpack((size_t)kPtr * kHid), pos(kPtr);
        std::vector<float> cmpos((size_t)kMem * kTok), tpos((size_t)kMask * kMem),
            tpw((size_t)kMem * kHid), tpb(kMem);
        for (int s = 0; s < kMask; s++) { maskmem[s].resize((size_t)kMem * kTok);
            for (size_t i = 0; i < maskmem[s].size(); i++) maskmem[s][i] = pat((int)i + s * 7); }
        for (size_t i = 0; i < objpack.size(); i++) objpack[i] = pat((int)i + 11);
        for (int p = 0; p < kPtr; p++) pos[p] = (p == 0) ? 45.f : (float)p;
        for (size_t i = 0; i < cmpos.size(); i++) cmpos[i] = pat((int)i + 3);
        for (size_t i = 0; i < tpos.size(); i++) tpos[i] = pat((int)i + 5);
        for (size_t i = 0; i < tpw.size(); i++) tpw[i] = pat((int)i + 9) * 0.01f;
        for (int i = 0; i < kMem; i++) tpb[i] = pat(i + 13);
        std::unique_ptr<GuardedFloats> gmm[kMask];
        for (int s = 0; s < kMask; s++) gmm[s].reset(new GuardedFloats(maskmem[s]));
        GuardedFloats gop(objpack), gpos(pos), gcm(cmpos), gtp(tpos), gtw(tpw), gtb(tpb);
        const float *mm_h[kMask]; for (int s = 0; s < kMask; s++) mm_h[s] = gmm[s]->data();
        const float *op_h[kPtr];  for (int p = 0; p < kPtr; p++) op_h[p] = gop.data() + (size_t)p * kHid;
        GuardedFloats ghm(std::vector<float>((size_t)kTotal * kMem)), ghp(std::vector<float>((size_t)kTotal * kMem));
        nvmm::MemConsts mc{gcm.data(), gtp.data(), gtw.data(), gtb.data()};
        nvmm::assemble_memory(mm_h, op_h, gpos.data(), mc, ghm.data(), ghp.data());
        const std::vector<float> hm = ghm.copy(), hp = ghp.copy();
        float *dmm[kMask]; for (int s = 0; s < kMask; s++) dmm[s] = dev(maskmem[s]);
        float *dptrs; cudaMalloc(&dptrs, sizeof(dmm)); cudaMemcpy(dptrs, dmm, sizeof(dmm), cudaMemcpyHostToDevice);
        float *dop = dev(objpack), *dpos = dev(pos), *dcm = dev(cmpos), *dtp = dev(tpos), *dtw = dev(tpw), *dtb = dev(tpb);
        float *dmem, *dmp; cudaMalloc(&dmem, hm.size() * sizeof(float)); cudaMalloc(&dmp, hp.size() * sizeof(float));
        nvmm::k_assemble_memory(reinterpret_cast<const float *const *>(dptrs), dop, dpos, dcm, dtp, dtw, dtb, dmem, dmp, kTok, s);
        cudaStreamSynchronize(s);
        auto gm = host(dmem, hm.size()), gp = host(dmp, hp.size());
        report("assemble.mem", maxabs(gm, hm), 0);
        report("assemble.pos", maxabs(gp, hp), kAssemblePosBound);
        for (int sl = 0; sl < kMask; sl++) cudaFree(dmm[sl]);
        cudaFree(dptrs); cudaFree(dop); cudaFree(dpos); cudaFree(dcm); cudaFree(dtp); cudaFree(dtw); cudaFree(dtb); cudaFree(dmem); cudaFree(dmp);
    }

    expect("consts_round_trip_keeps_names_shapes_and_values", consts_round_trip_keeps_names_shapes_and_values());
    expect("consts_unknown_name_is_null", consts_unknown_name_is_null());
    expect("consts_missing_file_is_reported", consts_missing_file_is_reported());
    expect("consts_rejects_wrong_magic", consts_rejects_wrong_magic());
    expect("consts_rejects_header_shorter_than_count", consts_rejects_header_shorter_than_count());
    expect("consts_rejects_count_beyond_tensors", consts_rejects_count_beyond_tensors());
    expect("consts_rejects_short_name", consts_rejects_short_name());
    expect("consts_rejects_missing_ndim", consts_rejects_missing_ndim());
    expect("consts_rejects_missing_dim", consts_rejects_missing_dim());
    expect("consts_rejects_data_shorter_than_shape", consts_rejects_data_shorter_than_shape());
    expect("memdims_match_sam2_512_crop", memdims_match_sam2_512_crop());
    expect("mask_box_reaches_all_four_edges", mask_box_reaches_all_four_edges());
    expect("mask_box_excludes_pixels_equal_to_thresh", mask_box_excludes_pixels_equal_to_thresh());
    expect("mask_box_single_pixel_in_column_zero_is_valid", mask_box_single_pixel_in_column_zero_is_valid());
    expect("mask_box_of_empty_mask_is_invalid", mask_box_of_empty_mask_is_invalid());
    expect("mlp3_relu_clamps_hidden_layers_only", mlp3_relu_clamps_hidden_layers_only());
    expect("transpose_covers_one_past_two_blocks", transpose_covers_one_past_two_blocks(s));
    expect("add_per_channel_covers_one_past_two_blocks", add_per_channel_covers_one_past_two_blocks(s));
    expect("sigmoid_writes_each_element_once", sigmoid_writes_each_element_once(s));
    expect("threshold_zero_maps_to_lo", threshold_zero_maps_to_lo(s));
    expect("bilinear_identity_reproduces_one_past_a_block", bilinear_identity_reproduces_one_past_a_block(s));
    expect("mask_bbox_ignores_zero_and_reaches_last_pixel", mask_bbox_ignores_zero_and_reaches_last_pixel(s));
    expect("assemble_small_tok_keeps_layout_and_tail", assemble_small_tok_keeps_layout_and_tail(s));
    expect("gmc_window_reads_pitched_rows_exactly", gmc_window_reads_pitched_rows_exactly(s));
    expect("gmc_cross_power_normalises_and_zeroes_at_cutoff", gmc_cross_power_normalises_and_zeroes_at_cutoff(s));

    std::printf("%s\n", ok ? "KERNEL_PARITY_PASS" : "KERNEL_PARITY_FAIL");
    return ok ? 0 : 1;
}
