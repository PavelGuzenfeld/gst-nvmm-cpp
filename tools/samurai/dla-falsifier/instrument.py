#!/usr/bin/env python3
"""Insert env-gated SAMURAI_TIMING probes into samurai_tracker.cpp; throwaway, idempotent via a marker.
Per frame: enc, prebox (steps 1-7), tail (8-9), cudaEvent-timed memenc_gpu, tail_bubble = tail - memenc_gpu.
The memenc anchor includes k_sigmoid_scale so it skips the identical seed-path block."""
import sys

p = sys.argv[1]
s = open(p).read()
if "SAMURAI_TIMING enc=" in s:
    print("already instrumented"); sys.exit(0)

def rep(old, new, s):
    if old not in s:
        print("ANCHOR NOT FOUND:\n" + old); sys.exit(2)
    return s.replace(old, new, 1)

s = rep("#include <vector>\n",
        "#include <vector>\n#include <chrono>\n", s)

s = rep("struct SamuraiTracker::Impl {",
        "static double g_prebox_ms = 0, g_tail_ms = 0, g_memenc_ms = -1;\n\n"
        "struct SamuraiTracker::Impl {", s)

s = rep("    frame_idx++;\n",
        "    frame_idx++;\n"
        "    const bool _tmg = std::getenv(\"SAMURAI_TIMING\") != nullptr;\n"
        "    std::chrono::steady_clock::time_point _tf0, _tbox;\n"
        "    static cudaEvent_t _evM0 = nullptr, _evM1 = nullptr;\n"
        "    if (_tmg) {\n"
        "        _tf0 = std::chrono::steady_clock::now();\n"
        "        if (!_evM0) { cudaEventCreate(&_evM0); cudaEventCreate(&_evM1); }\n"
        "    }\n", s)

s = rep("    out.target_id = 1;\n",
        "    out.target_id = 1;\n"
        "    if (_tmg) _tbox = std::chrono::steady_clock::now();\n", s)

s = rep(
    "    k_sigmoid_scale(d_high, d_mem_mask, HI * HI, 20.f, -10.f, stream);\n"
    "#endif\n"
    "    if (!mem_encoder->infer(stream)) { err = \"memory_encoder infer failed\"; return false; }\n"
    "    if (cudaStreamSynchronize(stream) != cudaSuccess) { err = \"memenc sync\"; return false; }\n",
    "    k_sigmoid_scale(d_high, d_mem_mask, HI * HI, 20.f, -10.f, stream);\n"
    "#endif\n"
    "    if (_tmg) cudaEventRecord(_evM0, stream);\n"
    "    if (!mem_encoder->infer(stream)) { err = \"memory_encoder infer failed\"; return false; }\n"
    "    if (_tmg) cudaEventRecord(_evM1, stream);\n"
    "    if (cudaStreamSynchronize(stream) != cudaSuccess) { err = \"memenc sync\"; return false; }\n"
    "    if (_tmg) { float _ms = 0; cudaEventElapsedTime(&_ms, _evM0, _evM1); g_memenc_ms = _ms; }\n",
    s)

s = rep('    GST_LOG("track f=%ld sel=%d obj=%.2f stable=%d box',
        "    if (_tmg) {\n"
        "        auto _tf1 = std::chrono::steady_clock::now();\n"
        "        using _D = std::chrono::duration<double, std::milli>;\n"
        "        g_prebox_ms = _D(_tbox - _tf0).count();\n"
        "        g_tail_ms   = _D(_tf1 - _tbox).count();\n"
        "    }\n"
        '    GST_LOG("track f=%ld sel=%d obj=%.2f stable=%d box', s)

s = rep(
    "    if (!impl_->run_encoder(frame, impl_->last, err)) {\n"
    "        GST_WARNING(\"track encoder failed: %s\", err.c_str());\n"
    "        return false;\n"
    "    }\n"
    "    if (!impl_->track_frame(out, err)) {\n"
    "        GST_WARNING(\"track_frame failed: %s\", err.c_str());\n"
    "        return false;\n"
    "    }\n"
    "    out.is_kf_only = false;\n"
    "    return true;\n",
    "    const bool _tmg = std::getenv(\"SAMURAI_TIMING\") != nullptr;\n"
    "    auto _te0 = std::chrono::steady_clock::now();\n"
    "    if (!impl_->run_encoder(frame, impl_->last, err)) {\n"
    "        GST_WARNING(\"track encoder failed: %s\", err.c_str());\n"
    "        return false;\n"
    "    }\n"
    "    auto _te1 = std::chrono::steady_clock::now();\n"
    "    if (!impl_->track_frame(out, err)) {\n"
    "        GST_WARNING(\"track_frame failed: %s\", err.c_str());\n"
    "        return false;\n"
    "    }\n"
    "    if (_tmg && g_memenc_ms >= 0) {\n"
    "        double _enc = std::chrono::duration<double, std::milli>(_te1 - _te0).count();\n"
    "        double _frame = _enc + g_prebox_ms + g_tail_ms;\n"
    "        double _bub = g_tail_ms - g_memenc_ms;\n"
    "        fprintf(stderr, \"SAMURAI_TIMING enc=%.3f prebox=%.3f tail=%.3f \"\n"
    "                \"memenc_gpu=%.3f tail_bubble=%.3f tail/frame=%.1f%% bubble/frame=%.1f%%\\n\",\n"
    "                _enc, g_prebox_ms, g_tail_ms, g_memenc_ms, _bub,\n"
    "                100.0 * g_tail_ms / _frame, 100.0 * _bub / _frame);\n"
    "        g_memenc_ms = -1;\n"
    "    }\n"
    "    out.is_kf_only = false;\n"
    "    return true;\n",
    s)

open(p, "w").write(s)
print("instrumented OK")
