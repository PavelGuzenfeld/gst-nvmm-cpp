# TensorRT cascade classifier (Phase 3.2). Shares the TrtEngine wrapper with
# nvmminfer (compiled in, not linked: the engine wrapper is header+one-cpp).
# secondary_cache.cpp is pure host and additionally unit-tested on x86 CI.
gstnvmmsecondaryinfer = library('gstnvmmsecondaryinfer',
  files(
    'gstnvmmsecondaryinfer.cpp',
    'roi_preprocess.cpp',
    'secondary_cache.cpp',
    '../nvmminfer/trt_engine.cpp',
  ),  # nvmm_class_meta.cpp lives in gst/common (CI-built + tested)
  include_directories : [
    config_inc,
    common_inc,
    include_directories('../nvmmalloc'),
    include_directories('../nvmminfer'),
  ],
  dependencies : [
    nvmm_common_dep, gstnvmmalloc_dep,
    gst_video_dep, gst_base_dep,
    trt_dep, nvbufsurftransform_dep,
  ],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)
