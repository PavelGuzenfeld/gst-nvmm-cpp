/// trt_dec_probe — validate TrtEngine::set_input_shape + the dynamic mask_decoder
/// binding in C++. Loads the Phase-A golden decoder inputs (.dat) for each engine
/// input, infers (resolving any -1 axis from the .dat element count), and dumps
/// the outputs as raw f32 .bin for offline cosine compare vs mask_decoder_expected.
///
/// Usage: trt_dec_probe <engine> <dat_dir> <dat_prefix> <dump_dir>
///   reads <dat_dir>/<dat_prefix>__<input>.dat for each engine input.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "trt_engine.hpp"

static std::vector<float> read_dat(const std::string &path, bool &ok)
{
    ok = false;
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "missing %s\n", path.c_str()); return {}; }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<float> v(n / sizeof(float));
    ok = std::fread(v.data(), 1, n, f) == (size_t)n;
    std::fclose(f);
    return v;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s engine dat_dir dat_prefix dump_dir\n", argv[0]);
        return 2;
    }
    const std::string engine = argv[1], dir = argv[2], prefix = argv[3], dump = argv[4];
    std::string err;
    auto eng = nvmm::TrtEngine::load_file(engine, err);
    if (!eng) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }

    std::vector<void *> to_free;
    // Inputs: read .dat, resolve dynamic axis, set shape, bind.
    for (const nvmm::TensorInfo &t : eng->tensors()) {
        if (!t.is_input) continue;
        bool ok = false;
        auto host = read_dat(dir + "/" + prefix + "__" + t.name + ".dat", ok);
        if (!ok) { std::fprintf(stderr, "read %s failed\n", t.name.c_str()); return 1; }
        // Resolve shape: replace a single -1 dim from the element count.
        std::vector<int64_t> dims(t.dims.nbDims);
        int64_t known = 1; int dyn = -1;
        for (int i = 0; i < t.dims.nbDims; i++) {
            dims[i] = t.dims.d[i];
            if (dims[i] < 0) dyn = i; else known *= dims[i];
        }
        if (dyn >= 0) dims[dyn] = (int64_t)host.size() / known;
        if (!eng->set_input_shape(t.name, dims)) {
            std::fprintf(stderr, "set_input_shape %s failed\n", t.name.c_str()); return 1;
        }
        void *d = nullptr;
        cudaMalloc(&d, host.size() * sizeof(float));
        cudaMemcpy(d, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
        eng->bind(t.name, d); to_free.push_back(d);
        std::printf("  in  %-12s elems=%zu shape=[", t.name.c_str(), host.size());
        for (auto v : dims) std::printf("%ld,", (long)v);
        std::printf("]\n");
    }
    // Outputs: alloc by engine volume, bind.
    std::vector<std::pair<std::string, std::pair<void *, size_t>>> outs;
    for (const nvmm::TensorInfo &t : eng->tensors()) {
        if (t.is_input) continue;
        void *d = nullptr;
        cudaMalloc(&d, t.bytes);
        eng->bind(t.name, d); to_free.push_back(d);
        outs.push_back({t.name, {d, t.bytes}});
    }
    cudaStream_t st; cudaStreamCreate(&st);
    if (!eng->infer(st)) { std::fprintf(stderr, "infer failed\n"); return 1; }
    cudaStreamSynchronize(st);
    for (auto &o : outs) {
        std::vector<char> host(o.second.second);
        cudaMemcpy(host.data(), o.second.first, o.second.second, cudaMemcpyDeviceToHost);
        std::string path = dump + "/" + o.first + ".bin";
        if (std::FILE *f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(host.data(), 1, host.size(), f); std::fclose(f);
        }
        std::printf("  out %-12s %zu bytes -> %s\n", o.first.c_str(), o.second.second, path.c_str());
    }
    for (void *p : to_free) cudaFree(p);
    std::printf("DEC_PROBE_OK\n");
    return 0;
}
