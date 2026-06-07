gstnvmmofa = library('gstnvmmofa',
  files(
    'gstnvmmofa.cpp',
    'gstnvmmflowstats.cpp',
    'nvmm_optical_flow_meta.cpp',
  ),
  include_directories : [
    config_inc,
    common_inc,
    include_directories('../nvmmalloc'),
  ],
  dependencies : [nvmm_common_dep, gstnvmmalloc_dep, gst_video_dep, gst_base_dep, vpi_dep],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)
