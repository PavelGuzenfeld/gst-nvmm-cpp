# JetPack 5 IPC backend (shm copy protocol).
# Built into a static library that sink + appsrc plugins link against.

ipc_backend_src = ['ipc_jp5.cpp']

ipc_backend_deps = [
  gst_dep, gst_base_dep, gst_video_dep,
  gstnvmmalloc_dep,
  nvmm_common_dep,
  cpp.find_library('rt', required : true),   # shm_open/shm_unlink
]
if have_nvbufsurface
  ipc_backend_deps += nvbufsurface_dep
endif
if nvbufsurftransform_dep.found()
  ipc_backend_deps += nvbufsurftransform_dep
endif

ipc_backend_lib = static_library(
  'nvmm_ipc_backend',
  ipc_backend_src,
  include_directories : [config_inc, common_inc, include_directories('../nvmmalloc')],
  dependencies : ipc_backend_deps,
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

ipc_backend_dep = declare_dependency(
  link_with : ipc_backend_lib,
  include_directories : [common_inc],
  dependencies : ipc_backend_deps,
)
