project('gst-nvmm-cpp', ['c', 'cpp'],
  version : '1.2.0',
  meson_version : '>= 0.62.0',
  license : 'LGPL-2.1-or-later',
  default_options : [
    'cpp_std=c++14',
    'c_std=c11',
    'warning_level=2',
    'werror=true',
    'buildtype=debugoptimized',
  ],
)

gst_dep = dependency('gstreamer-1.0', version : '>= 1.16.0')
gst_base_dep = dependency('gstreamer-base-1.0', version : '>= 1.16.0')
gst_video_dep = dependency('gstreamer-video-1.0', version : '>= 1.16.0')
gst_allocators_dep = dependency('gstreamer-allocators-1.0', version : '>= 1.16.0')

# NVIDIA Jetson libraries — found via pkg-config or manual paths
# NvBufSurface is part of the NVIDIA Multimedia API
cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

nvmm_lib_dirs = [
  '/usr/lib/aarch64-linux-gnu/tegra',
  '/usr/lib/aarch64-linux-gnu/nvidia',
  '/opt/nvidia/deepstream/deepstream/lib',
  '/usr/lib/nvidia',
  '/usr/lib/aarch64-linux-gnu',
]

nvbufsurface_dep = dependency('nvbufsurface', required : false)
if not nvbufsurface_dep.found()
  # Fallback: search standard Jetson paths
  nvbufsurface_lib = cc.find_library('nvbufsurface',
    dirs : nvmm_lib_dirs,
    required : false,
  )
  if nvbufsurface_lib.found()
    fsmod = import('fs')
    nvmm_inc_candidates = [
      '/usr/src/jetson_multimedia_api/include',
      '/usr/include/nvbufsurface',
      '/opt/nvidia/deepstream/deepstream/sources/includes',
    ]
    nvmm_inc_dirs = []
    foreach d : nvmm_inc_candidates
      if fsmod.is_dir(d)
        nvmm_inc_dirs += d
      endif
    endforeach
    if nvmm_inc_dirs.length() > 0
      nvbufsurface_inc = include_directories(nvmm_inc_dirs, is_system : true)
      nvbufsurface_dep = declare_dependency(
        dependencies : nvbufsurface_lib,
        include_directories : nvbufsurface_inc,
      )
    endif
  endif
endif

nvbufsurftransform_dep = dependency('nvbufsurftransform', required : false)
if not nvbufsurftransform_dep.found()
  nvbufsurftransform_lib = cc.find_library('nvbufsurftransform',
    dirs : nvmm_lib_dirs,
    required : false,
  )
  if nvbufsurftransform_lib.found()
    nvbufsurftransform_dep = declare_dependency(
      dependencies : nvbufsurftransform_lib,
    )
  endif
endif

# Feature flags
have_nvbufsurface = nvbufsurface_dep.found()
have_nvbufsurftransform = nvbufsurftransform_dep.found()

# JetPack version detection.
#
# We do NOT rely on the L4T release file — that only works when configuring
# on a Jetson. Instead, probe the nvbufsurface.h header for a JP6-only
# symbol (NvBufSurfaceMapParams). This also works when cross-compiling
# against JP6 headers in a Docker build on an x86 host.
jetpack_version = 'unknown'
have_jp6_api = false
if have_nvbufsurface
  nvmm_hdr_dirs = []
  fsmod = import('fs')
  foreach d : ['/usr/src/jetson_multimedia_api/include',
               '/usr/include/nvbufsurface',
               '/opt/nvidia/deepstream/deepstream/sources/includes']
    if fsmod.is_dir(d)
      nvmm_hdr_dirs += '-I' + d
    endif
  endforeach
  have_jp6_api = cpp.has_header_symbol(
    'nvbufsurface.h', 'NvBufSurfaceMapParams',
    args : nvmm_hdr_dirs,
    required : false)
  jetpack_version = have_jp6_api ? '6' : '5'
else
  # Mock build — always exercises the JP6 backend (mock headers carry stubs
  # for every JP6 API the backend calls).
  have_jp6_api = true
  jetpack_version = 'mock-jp6'
endif

message('selected IPC backend: @0@  (JetPack: @1@)'.format(
  have_jp6_api ? 'jp6 (pool + SCM_RIGHTS)' : 'jp5 (shm-copy)',
  jetpack_version))

# NvSciBuf support (JP6+ only)
have_nvscibuf = false
if jetpack_version == '6'
  nvscibuf_dep = dependency('nvscibuf', required : false)
  if not nvscibuf_dep.found()
    nvscibuf_lib = cc.find_library('nvscibuf', dirs : nvmm_lib_dirs, required : false)
    if nvscibuf_lib.found()
      nvscibuf_dep = declare_dependency(dependencies : nvscibuf_lib)
      have_nvscibuf = true
    endif
  else
    have_nvscibuf = true
  endif
endif

# Configuration header
conf = configuration_data()
conf.set10('HAVE_NVBUFSURFACE', have_nvbufsurface)
conf.set10('HAVE_NVBUFSURFTRANSFORM', have_nvbufsurftransform)
conf.set10('HAVE_NVSCIBUF', have_nvscibuf)
conf.set_quoted('JETPACK_VERSION', jetpack_version)
conf.set_quoted('PACKAGE_VERSION', meson.project_version())

configure_file(
  output : 'config.h',
  configuration : conf,
)

config_inc = include_directories('.')
common_inc = include_directories('gst/common')

# When NVMM libs not available, build mock versions for testing
build_mock = not have_nvbufsurface
if build_mock
  message('NvBufSurface not found — building with mock API for host-side testing')
endif

subdir('gst/common')
subdir('gst/nvmmalloc')   # provides gstnvmmalloc_dep, consumed by jp5 backend

# IPC backend — exactly one of these subdirs is compiled, picked by JP detection.
# The subdir defines `ipc_backend_dep` which sink/appsrc depend on.
if have_jp6_api
  subdir('gst/ipc_jp6')
else
  subdir('gst/ipc_jp5')
endif

subdir('gst/nvmmconvert')
subdir('gst/nvmmsink')
subdir('gst/nvmmappsrc')
subdir('tests')
subdir('benchmarks')
