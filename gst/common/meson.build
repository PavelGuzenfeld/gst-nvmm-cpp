common_sources = []
common_deps = [gst_dep, gst_base_dep, gst_video_dep, gst_allocators_dep]
common_cpp_args = []

common_sources += files('nvmm_buffer.cpp', 'nvmm_transform.cpp')
common_sources += files('nvmm_optical_flow_meta.cpp')
common_sources += files('nvmm_det_meta.cpp')
common_sources += files('nvmm_motion.cpp', 'nvmm_motion_meta.cpp')
common_sources += files('nvmm_class_meta.cpp')
common_sources += files('nvmm_track_meta.cpp')
common_sources += files('kalman_box.cpp')
if build_mock
  mock_inc = include_directories('.')
else
  common_deps += [nvbufsurface_dep, nvbufsurftransform_dep]
endif
if build_mock
  common_cpp_args += ['-DNVMM_MOCK_API']
endif
if have_deepstream_meta
  common_cpp_args += ['-DNVMM_DEEPSTREAM_META']
  common_deps += [nvds_meta_dep]
endif

nvmm_common_lib = shared_library('nvmm_common',
  common_sources,
  include_directories : [config_inc, common_inc],
  dependencies : common_deps,
  cpp_args : common_cpp_args,
  install : true,
)

nvmm_common_dep = declare_dependency(
  link_with : nvmm_common_lib,
  include_directories : [config_inc, common_inc],
  dependencies : common_deps,
)
