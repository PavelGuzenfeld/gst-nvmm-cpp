#include "trt_engine.hpp"

#include <cstdio>
#include <fstream>

namespace nvmm {
namespace {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity s, const char *msg) noexcept override {
        if (s <= Severity::kWARNING)
            std::fprintf(stderr, "[nvmminfer/TRT] %s\n", msg);
    }
};
Logger g_logger;

size_t dtype_size(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kBOOL:  return 1;
        case nvinfer1::DataType::kUINT8: return 1;
#if NV_TENSORRT_MAJOR >= 9
        case nvinfer1::DataType::kINT64: return 8;
#endif
        default: return 0;
    }
}

}  // namespace

const char *dtype_str(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return "f32";
        case nvinfer1::DataType::kHALF:  return "f16";
        case nvinfer1::DataType::kINT32: return "i32";
        case nvinfer1::DataType::kINT8:  return "i8";
        case nvinfer1::DataType::kBOOL:  return "bool";
        case nvinfer1::DataType::kUINT8: return "u8";
        default: return "?";
    }
}

std::string dims_str(const nvinfer1::Dims &d) {
    std::string s;
    for (int i = 0; i < d.nbDims; i++) {
        if (i) s += "x";
        s += std::to_string(d.d[i]);
    }
    return s.empty() ? "scalar" : s;
}

std::unique_ptr<TrtEngine> TrtEngine::load_file(const std::string &path, std::string &err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { err = "cannot open engine file: " + path; return nullptr; }
    const std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(static_cast<size_t>(n));
    if (n <= 0 || !f.read(blob.data(), n)) {
        err = "failed to read engine file: " + path;
        return nullptr;
    }

    std::unique_ptr<TrtEngine> e(new TrtEngine());
    e->runtime_.reset(nvinfer1::createInferRuntime(g_logger));
    if (!e->runtime_) { err = "createInferRuntime failed"; return nullptr; }

    e->engine_.reset(e->runtime_->deserializeCudaEngine(blob.data(), static_cast<size_t>(n)));
    if (!e->engine_) {
        err = "deserializeCudaEngine failed (engine built for a different "
              "TensorRT/GPU/DLA than this device?)";
        return nullptr;
    }

    e->ctx_.reset(e->engine_->createExecutionContext());
    if (!e->ctx_) { err = "createExecutionContext failed"; return nullptr; }

    const int nt = e->engine_->getNbIOTensors();
    for (int i = 0; i < nt; i++) {
        const char *name = e->engine_->getIOTensorName(i);
        TensorInfo ti;
        ti.name     = name;
        ti.is_input = e->engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        ti.dtype    = e->engine_->getTensorDataType(name);
        ti.dims     = e->engine_->getTensorShape(name);
        int64_t vol = 1;
        for (int d = 0; d < ti.dims.nbDims; d++) {
            const int64_t dim = ti.dims.d[d];
            vol *= (dim > 0 ? dim : 1);
        }
        ti.volume = vol;
        ti.bytes  = static_cast<size_t>(vol) * dtype_size(ti.dtype);
        e->tensors_.push_back(std::move(ti));
    }
    return e;
}

const TensorInfo *TrtEngine::input0() const {
    for (const auto &t : tensors_)
        if (t.is_input) return &t;
    return nullptr;
}

const TensorInfo *TrtEngine::output0() const {
    for (const auto &t : tensors_)
        if (!t.is_input) return &t;
    return nullptr;
}

bool TrtEngine::bind(const std::string &name, void *device_ptr) {
    return ctx_->setTensorAddress(name.c_str(), device_ptr);
}

bool TrtEngine::infer(cudaStream_t stream) {
    return ctx_->enqueueV3(stream);
}

TrtEngine::~TrtEngine() = default;

}  // namespace nvmm
