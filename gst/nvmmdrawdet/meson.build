gstnvmmdrawdet = library('gstnvmmdrawdet',
  files('gstnvmmdrawdet.cpp'),  # nvmm_det_meta.cpp lives in gst/common
  include_directories : [
    config_inc,
    common_inc,
    include_directories('../nvmmalloc'),
  ],
  dependencies : [
    nvmm_common_dep, gstnvmmalloc_dep,
    gst_video_dep, gst_base_dep,
    trt_dep, nvbufsurftransform_dep,   # trt_dep pulls cudart (EGL interop)
  ],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)
