# JetPack 6 IPC backend (pool + SCM_RIGHTS protocol).
# Built into a static library that sink + appsrc plugins link against.

ipc_backend_src = ['ipc_jp6.cpp']

ipc_backend_deps = [
  gst_dep, gst_base_dep, gst_video_dep,
  nvmm_common_dep,
]
if have_nvbufsurface
  ipc_backend_deps += nvbufsurface_dep
endif

thread_dep = dependency('threads')

ipc_backend_lib = static_library(
  'nvmm_ipc_backend',
  ipc_backend_src,
  include_directories : [config_inc, common_inc],
  dependencies : ipc_backend_deps + [thread_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

ipc_backend_dep = declare_dependency(
  link_with : ipc_backend_lib,
  include_directories : [common_inc],
  dependencies : ipc_backend_deps + [thread_dep],
)
