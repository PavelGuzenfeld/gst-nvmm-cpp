#ifndef NVMM_TRT_TEST_ENGINE_H
#define NVMM_TRT_TEST_ENGINE_H

#include <NvInfer.h>
#include <glib.h>
#include <glib/gstdio.h>

#include <cstdio>
#include <unistd.h>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class BuildLogger : public nvinfer1::ILogger {
public:
    void log(Severity s, const char *msg) noexcept override
    {
        if (s <= Severity::kERROR) std::fprintf(stderr, "[trt-build] %s\n", msg);
    }
};

/// Removes the serialized engine when the test is done with it.
struct EngineFile {
    std::string path;
    explicit EngineFile(std::string p) : path(std::move(p)) {}
    EngineFile(const EngineFile &) = delete;
    ~EngineFile() { g_unlink(path.c_str()); }
};

/// Builds a tiny FP32 network on this device, so no model file is needed.
class NetBuilder {
public:
    NetBuilder()
        : builder_(nvinfer1::createInferBuilder(logger())),
          net_(builder_->createNetworkV2(0U)) {}

    nvinfer1::INetworkDefinition &net() { return *net_; }

    nvinfer1::ITensor *input(nvinfer1::Dims d)
    {
        return net_->addInput("images", nvinfer1::DataType::kFLOAT, d);
    }

    nvinfer1::ITensor *constant(nvinfer1::Dims d, std::vector<float> v)
    {
        weights_.push_back(std::move(v));
        const std::vector<float> &w = weights_.back();
        return net_->addConstant(d, {nvinfer1::DataType::kFLOAT, w.data(),
                                     (int64_t)w.size()})->getOutput(0);
    }

    nvinfer1::ITensor *reduce_max(nvinfer1::ITensor *t, uint32_t axes, bool keep)
    {
        return net_->addReduce(*t, nvinfer1::ReduceOperation::kMAX, axes, keep)->getOutput(0);
    }

    nvinfer1::ITensor *reshape(nvinfer1::ITensor *t, nvinfer1::Dims d)
    {
        nvinfer1::IShuffleLayer *s = net_->addShuffle(*t);
        s->setReshapeDimensions(d);
        return s->getOutput(0);
    }

    nvinfer1::ITensor *binary(nvinfer1::ITensor *a, nvinfer1::ITensor *b,
                              nvinfer1::ElementWiseOperation op)
    {
        return net_->addElementWise(*a, *b, op)->getOutput(0);
    }

    /// A constant of shape `d` that still reads the input, so it is never pruned.
    nvinfer1::ITensor *constant_fed_by(nvinfer1::ITensor *in, nvinfer1::Dims d)
    {
        nvinfer1::Dims ones = d;
        for (int i = 0; i < ones.nbDims; i++) ones.d[i] = 1;
        nvinfer1::ITensor *m = reshape(reduce_max(in, 0xEu, true), ones);
        return binary(constant(d, std::vector<float>((size_t)volume(d), 0.f)), m,
                      nvinfer1::ElementWiseOperation::kSUM);
    }

    void output(nvinfer1::ITensor *t, const char *name)
    {
        t->setName(name);
        net_->markOutput(*t);
    }

    std::unique_ptr<EngineFile> save()
    {
        std::unique_ptr<nvinfer1::IBuilderConfig> cfg(builder_->createBuilderConfig());
        cfg->setBuilderOptimizationLevel(0);
        std::unique_ptr<nvinfer1::IHostMemory> blob(
            builder_->buildSerializedNetwork(*net_, *cfg));
        if (!blob) throw std::runtime_error("buildSerializedNetwork failed");
        gchar *path = nullptr;
        const gint fd = g_file_open_tmp("nvmm-test-XXXXXX.engine", &path, nullptr);
        if (fd < 0) throw std::runtime_error("g_file_open_tmp failed");
        close(fd);
        if (!g_file_set_contents(path, static_cast<const gchar *>(blob->data()),
                                 (gssize)blob->size(), nullptr))
            throw std::runtime_error("engine write failed");
        std::unique_ptr<EngineFile> f(new EngineFile(path));
        g_free(path);
        return f;
    }

private:
    static int64_t volume(nvinfer1::Dims d)
    {
        int64_t v = 1;
        for (int i = 0; i < d.nbDims; i++) v *= d.d[i];
        return v;
    }
    static BuildLogger &logger()
    {
        static BuildLogger l;
        return l;
    }

    std::unique_ptr<nvinfer1::IBuilder> builder_;
    std::unique_ptr<nvinfer1::INetworkDefinition> net_;
    std::list<std::vector<float>> weights_;
};

#endif
