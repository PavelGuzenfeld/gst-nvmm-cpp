rt_dep = cpp.find_library('rt', required : true)
thread_dep = dependency('threads')
cudart_dep = cpp.find_library('cudart', dirs : ['/usr/local/cuda/lib64'], required : false)

nvmmsink_inc = [
  config_inc,
  common_inc,
  include_directories('../nvmmalloc'),
]

# CUDA headers only exist on a real Jetson/CUDA host. include_directories()
# fails at configure time if the path is missing, so add it only when present
# (mock builds don't include cuda_runtime.h at all).
fsmod = import('fs')
if fsmod.is_dir('/usr/local/cuda/include')
  nvmmsink_inc += include_directories('/usr/local/cuda/include')
endif

gstnvmmsink = library('gstnvmmsink',
  files('gstnvmmsink.cpp'),
  include_directories : nvmmsink_inc,
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
