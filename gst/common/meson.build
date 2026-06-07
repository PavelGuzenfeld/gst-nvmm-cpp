common_sources = []
common_deps = [gst_dep, gst_base_dep, gst_video_dep, gst_allocators_dep]

# Single source for both targets: nvmm_buffer.cpp / nvmm_transform.cpp pick the
# mock vs real NvBufSurface header via NVMM_MOCK_API (see their #include guards).
common_sources += files('nvmm_buffer.cpp', 'nvmm_transform.cpp')
# Optical-flow metadata — pure GStreamer (no NVMM/VPI), so it builds and is
# unit-tested everywhere, including the x86 mock/CI build. Consumed by nvmmofa.
common_sources += files('nvmm_optical_flow_meta.cpp')
if build_mock
  mock_inc = include_directories('.')
else
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
