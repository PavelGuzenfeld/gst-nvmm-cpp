gstnvmminfer = library('gstnvmminfer',
  files(
    'gstnvmminfer.cpp',
    'trt_engine.cpp',
  ),  # nvmm_det_meta.cpp lives in gst/common (CI-built + tested)
  include_directories : [
    config_inc,
    common_inc,
    include_directories('../nvmmalloc'),
  ],
  dependencies : [
    nvmm_common_dep, gstnvmmalloc_dep,
    gst_video_dep, gst_base_dep,
    trt_dep, nvbufsurftransform_dep,
  ],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)
