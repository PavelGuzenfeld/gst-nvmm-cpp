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
  )
  benchmark('analytics_cuda_operations', bench_analytics_cuda, protocol : 'exitcode',
    timeout : 600)
endif
