gstnvmmalloc = library('gstnvmmalloc',
  files('gstnvmmallocator.cpp', 'plugin.cpp'),
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep, gst_allocators_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)

gstnvmmalloc_dep = declare_dependency(
  link_with : gstnvmmalloc,
  include_directories : [config_inc, common_inc, include_directories('.')],
  dependencies : [nvmm_common_dep, gst_allocators_dep],
)
