bench_nvmm = executable('bench_nvmm',
  'bench_nvmm.cpp',
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

benchmark('nvmm_operations', bench_nvmm, protocol : 'exitcode')
