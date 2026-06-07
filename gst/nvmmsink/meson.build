rt_dep = cpp.find_library('rt', required : true)
thread_dep = dependency('threads')

nvmmsink_deps = [nvmm_common_dep, gstnvmmalloc_dep, gst_video_dep, rt_dep, thread_dep]
if have_deepstream_meta
  # Pulls the NVMM_DEEPSTREAM_META define + nvdsmeta/gstnvdsmeta headers + libs so
  # the producer can read NvDsBatchMeta from incoming buffers.
  nvmmsink_deps += [nvds_meta_dep]
endif

gstnvmmsink = library('gstnvmmsink',
  files('gstnvmmsink.cpp'),
  include_directories : [config_inc, common_inc, include_directories('../nvmmalloc')],
  dependencies : nvmmsink_deps,
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)

gstnvmmsink_dep = declare_dependency(
  link_with : gstnvmmsink,
  include_directories : [config_inc, common_inc, include_directories('.')],
  dependencies : [nvmm_common_dep, gstnvmmalloc_dep, gst_video_dep, rt_dep],
)
