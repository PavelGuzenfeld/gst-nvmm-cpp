common_sources = []
common_deps = [gst_dep, gst_base_dep, gst_video_dep, gst_allocators_dep]
common_cpp_args = []

# Single source for both targets: nvmm_buffer.cpp / nvmm_transform.cpp pick the
# mock vs real NvBufSurface header via NVMM_MOCK_API (see their #include guards).
common_sources += files('nvmm_buffer.cpp', 'nvmm_transform.cpp')
# Optical-flow metadata — pure GStreamer (no NVMM/VPI), so it builds and is
# unit-tested everywhere, including the x86 mock/CI build. Consumed by nvmmofa.
common_sources += files('nvmm_optical_flow_meta.cpp')
# nvmm_det_meta.cpp is the DeepStream-free GstNvmmDetMeta carrying detections
# across the IPC boundary (the optional NvDs bridge inside it is #ifdef-gated).
common_sources += files('nvmm_det_meta.cpp')
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

# Shared (not static) so the GstMeta registrations live in ONE copy per process.
# A static lib is linked into every plugin .so, so each gets its own copy of the
# meta api_get_type() + its static GType cache; two plugins first-touching the
# same meta concurrently (e.g. nvmmofa producing + nvmmfusion consuming the flow
# meta on different threads) then race the process-global type registration and
# deadlock. One shared copy removes the duplication.
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
