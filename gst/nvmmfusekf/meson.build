# Pure-host element (no CUDA/TRT/VIC): master constant-velocity Kalman filter
# fusing SAMURAI track meta + YOLO det meta. Builds on every target.
gstnvmmfusekf = library('gstnvmmfusekf',
  files('gstnvmmfusekf.cpp'),
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep, gst_base_dep],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)
