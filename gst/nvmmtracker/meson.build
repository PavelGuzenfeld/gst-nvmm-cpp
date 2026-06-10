# Pure-host element (no CUDA/TRT/VIC): tracks GstNvmmDetMeta detections by IOU.
# Builds on every target, including the x86 mock CI build.
gstnvmmtracker = library('gstnvmmtracker',
  files('gstnvmmtracker.cpp', 'tracker.cpp'),
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep, gst_base_dep],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)
