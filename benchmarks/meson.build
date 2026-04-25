bench_nvmm = executable('bench_nvmm',
  'bench_nvmm.cpp',
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

benchmark('nvmm_operations', bench_nvmm, protocol : 'exitcode')

bench_ipc_backend = executable('bench_ipc_backend',
  'bench_ipc_backend.cpp',
  include_directories : [
    config_inc, common_inc,
    include_directories('../gst/nvmmalloc'),
  ],
  dependencies : [
    ipc_backend_dep, gstnvmmalloc_dep, gst_dep, gst_video_dep,
    dependency('threads'),
  ],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

benchmark('ipc_backend_throughput', bench_ipc_backend,
  protocol : 'exitcode', timeout : 60)
