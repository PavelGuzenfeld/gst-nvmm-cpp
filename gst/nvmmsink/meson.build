rt_dep = cpp.find_library('rt', required : true)
thread_dep = dependency('threads')
cudart_dep = cpp.find_library('cudart', dirs : ['/usr/local/cuda/lib64'], required : false)

gstnvmmsink = library('gstnvmmsink',
  files('gstnvmmsink.cpp'),
  include_directories : [
    config_inc,
    common_inc,
    include_directories('../nvmmalloc'),
    include_directories('/usr/local/cuda/include'),
  ],
  dependencies : [nvmm_common_dep, gstnvmmalloc_dep, gst_video_dep, rt_dep, thread_dep, cudart_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)

gstnvmmsink_dep = declare_dependency(
  link_with : gstnvmmsink,
  include_directories : [config_inc, common_inc, include_directories('.')],
  dependencies : [nvmm_common_dep, gstnvmmalloc_dep, gst_video_dep, rt_dep],
)
