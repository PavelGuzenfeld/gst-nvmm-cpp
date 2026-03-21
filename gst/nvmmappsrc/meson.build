gstnvmmappsrc = library('gstnvmmappsrc',
  files('gstnvmmappsrc.cpp'),
  include_directories : [
    config_inc,
    common_inc,
    include_directories('../nvmmalloc'),
  ],
  dependencies : [nvmm_common_dep, gstnvmmalloc_dep, gst_video_dep, rt_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)

gstnvmmappsrc_dep = declare_dependency(
  link_with : gstnvmmappsrc,
  include_directories : [config_inc, common_inc, include_directories('.')],
  dependencies : [nvmm_common_dep, gstnvmmalloc_dep, gst_video_dep, rt_dep],
)
