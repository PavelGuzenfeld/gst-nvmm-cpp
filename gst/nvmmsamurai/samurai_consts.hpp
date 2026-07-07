/// samurai_consts.hpp — loader for the packed SAMURAI out-of-engine constants
/// (samurai_consts.bin, produced by pack_consts.py). Header-only; host-side only.
///
/// .bin format (little-endian):
///   u32 magic 0x5341434E ('SACN'), u32 count
///   per tensor: u32 name_len, name bytes, u32 ndim, ndim*u32 dims, f32 data
#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace nvmm {

struct ConstTensor {
    std::vector<int>   shape;
    std::vector<float> data;   // row-major
    size_t count() const { return data.size(); }
};

class SamuraiConsts {
public:
    bool load(const std::string &path, std::string &err)
    {
        std::FILE *f = std::fopen(path.c_str(), "rb");
        if (!f) { err = "cannot open consts: " + path; return false; }
        auto rd_u32 = [&](uint32_t &v) { return std::fread(&v, 4, 1, f) == 1; };
        uint32_t magic = 0, count = 0;
        if (!rd_u32(magic) || !rd_u32(count) || magic != 0x5341434Eu) {
            err = "bad consts magic/header"; std::fclose(f); return false;
        }
        for (uint32_t i = 0; i < count; i++) {
            uint32_t nlen = 0;
            if (!rd_u32(nlen)) { err = "truncated name len"; std::fclose(f); return false; }
            std::string name(nlen, '\0');
            if (nlen && std::fread(&name[0], 1, nlen, f) != nlen) { err = "truncated name"; std::fclose(f); return false; }
            uint32_t ndim = 0;
            if (!rd_u32(ndim)) { err = "truncated ndim"; std::fclose(f); return false; }
            ConstTensor t;
            size_t n = 1;
            t.shape.resize(ndim);
            for (uint32_t d = 0; d < ndim; d++) {
                uint32_t dim = 0;
                if (!rd_u32(dim)) { err = "truncated dim"; std::fclose(f); return false; }
                t.shape[d] = (int)dim; n *= dim;
            }
            t.data.resize(n);
            if (std::fread(t.data.data(), sizeof(float), n, f) != n) {
                err = "truncated data for " + name; std::fclose(f); return false;
            }
            tensors_[name] = std::move(t);
        }
        std::fclose(f);
        return true;
    }

    const ConstTensor *get(const std::string &name) const
    {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }
    const float *data(const std::string &name) const
    {
        const ConstTensor *t = get(name);
        return t ? t->data.data() : nullptr;
    }
    size_t size() const { return tensors_.size(); }

private:
    std::map<std::string, ConstTensor> tensors_;
};

}  // namespace nvmm
