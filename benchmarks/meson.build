bench_nvmm = executable('bench_nvmm',
  'bench_nvmm.cpp',
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

benchmark('nvmm_operations', bench_nvmm, protocol : 'exitcode')

# OpenCV vs hand-rolled (fused) analytics throughput — needs OpenCV, so it lives
# in the golden lane. Emits the same CSV convention as bench_nvmm. Shares its
# scene generator (bench_scene.h) with bench_analytics_cuda below, and its
# img::Image<->cv::Mat conversion (golden_util.h, hence tests_inc) with the
# golden-comparison tests.
if have_analytics_golden
  bench_analytics = executable('bench_analytics',
    'bench_analytics.cpp',
    include_directories : [analytics_inc, tests_inc],
    dependencies : [opencv4_dep],
  )
  benchmark('analytics_operations', bench_analytics, protocol : 'exitcode',
    timeout : 600)
endif

# Fused host vs CUDA low_texture_motion (upload/download included) — real-GPU,
# no OpenCV. Shares its scene generator (bench_scene.h) with bench_analytics above.
if have_analytics_cuda
  bench_analytics_cuda = executable('bench_analytics_cuda',
    ['bench_analytics_cuda.cpp', '../analytics/analytics_kernels.cu'],
    include_directories : [analytics_inc],
    dependencies : [analytics_cuda_dep],
    cuda_args : analytics_cuda_args,
    # Link via g++, not nvcc: meson's CUDA linker for a mixed C++/CUDA target
    # emits -Wl,-rpath unwrapped and nvcc rejects it. cudart arrives via
    # analytics_cuda_dep; the kernels use no relocatable device code.
    link_language : 'cpp',
  )
  benchmark('analytics_cuda_operations', bench_analytics_cuda, protocol : 'exitcode',
    timeout : 600)
endif

# GMC estimator micro-bench (ncc / fft-cpu / fft-cuda / pva): us/frame + shift
# error on synthetic known-shift patches, isolated from the pipeline. Needs nvcc
# for the GMC CUDA kernels (samurai_kernels.cu); VPI enables the fft-cuda + pva
# backends. Jetson-only (real CUDA/VPI); also gates numeric correctness (exitcode).
if have_tensorrt
  bench_gmc_deps = [trt_dep]
  bench_gmc_cpp_args = ['-Wall', '-Wextra', '-Werror', '-Wshadow']
  if have_vpi
    bench_gmc_deps += vpi_dep
    bench_gmc_cpp_args += '-DNVMM_HAVE_VPI'
  endif
  bench_gmc = executable('bench_gmc',
    ['bench_gmc.cpp', '../gst/nvmmsamurai/samurai_kernels.cu'],
    include_directories : [common_inc, include_directories('../gst/nvmmsamurai')],
    dependencies : bench_gmc_deps,
    cuda_args : samurai_cuda_base_args,
    cpp_args : bench_gmc_cpp_args,
    link_language : 'cpp',
  )
  benchmark('gmc_estimators', bench_gmc, protocol : 'exitcode', timeout : 120)
endif
