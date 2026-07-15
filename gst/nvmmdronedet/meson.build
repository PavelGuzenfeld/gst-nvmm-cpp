# nvmmdronedet — autonomous drone-seed gate (YOLO ∩ independent motion). Uses the
# shared XFeat+LightGlue matcher (TRT) + the TrtEngine wrapper (compiled in, not
# linked), like nvmmsamurai. Gated on TRT + real NVMM (Jetson-only) with the other
# inference-graph elements.
gstnvmmdronedet = library('gstnvmmdronedet',
  files(
    'gstnvmmdronedet.cpp',
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
    trt_dep, nvbufsurftransform_dep,   # trt_dep brings nvinfer + cudart + NPP
  ],
  cpp_args : ['-Werror'],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)
