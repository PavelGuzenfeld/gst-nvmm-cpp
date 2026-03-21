common_sources = []
common_deps = [gst_dep, gst_base_dep, gst_video_dep, gst_allocators_dep]

if build_mock
  common_sources += files('nvmm_buffer_mock.cpp', 'nvmm_transform_mock.cpp')
  mock_inc = include_directories('.')
else
  common_sources += files('nvmm_buffer.cpp', 'nvmm_transform.cpp')
  common_deps += [nvbufsurface_dep, nvbufsurftransform_dep]
endif

nvmm_common_lib = static_library('nvmm_common',
  common_sources,
  include_directories : [config_inc, common_inc],
  dependencies : common_deps,
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

nvmm_common_dep = declare_dependency(
  link_with : nvmm_common_lib,
  include_directories : [config_inc, common_inc],
  dependencies : common_deps,
)
