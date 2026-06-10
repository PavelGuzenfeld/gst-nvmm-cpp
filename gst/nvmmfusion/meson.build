# Pure-host aggregator (no CUDA/TRT/VIC): joins the detector + optical-flow
# branches by PTS and unions their metas. Builds on every target, incl. x86 CI.
gstnvmmfusion = library('gstnvmmfusion',
  files('gstnvmmfusion.cpp'),
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep, gst_base_dep],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)
