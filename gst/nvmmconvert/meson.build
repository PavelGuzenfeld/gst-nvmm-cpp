gstnvmmconvert = library('gstnvmmconvert',
  files('gstnvmmconvert.cpp', 'plugin.cpp'),
  include_directories : [
    config_inc,
    common_inc,
    include_directories('../nvmmalloc'),
  ],
  dependencies : [nvmm_common_dep, gstnvmmalloc_dep, gst_video_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)
