gstnvmmdetgate = library('gstnvmmdetgate',
  files(
    'gstnvmmdetgate.cpp',
    '../common/xfeat_matcher.cpp',
    '../nvmminfer/trt_engine.cpp',
  ),
  include_directories : [
    config_inc,
    common_inc,
    include_directories('../nvmmalloc'),
    include_directories('../nvmminfer'),
  ],
  dependencies : [
    nvmm_common_dep, gstnvmmalloc_dep,
    gst_base_dep,
    trt_dep, nvbufsurftransform_dep,
  ],
  cpp_args : ['-Werror'],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)
