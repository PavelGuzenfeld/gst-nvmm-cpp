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
    # NOTE: do NOT include /opt/nvidia/deepstream/.../sources/includes here.
    # DeepStream <= 6.2 ships an older nvbufsurface.h pinned to JP 5.0.x and
    # missing NvBufSurfaceImport. If both DeepStream and jetson-multimedia-api
    # are installed, DeepStream's stale header silently wins on systems where
    # something else added it to the search path.
    nvmm_inc_candidates = [
      '/usr/src/jetson_multimedia_api/include',
      '/usr/include/nvbufsurface',
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

# NVMM cross-process import API check.
#
# Pool + SCM_RIGHTS IPC requires NvBufSurfaceImport / NvBufSurfaceMapParams /
# NvBufSurfaceGetMapParams. These ship in:
#   - L4T R35.3.1 (JetPack 5.1.1, March 2023) and later for the JP5 line
#   - L4T R36.0  (JetPack 6.0)                  and later for the JP6 line
#
# The earlier L4T 35.x revisions (R35.2.1 = JP 5.0.x) do NOT have the
# import API and cannot do cross-process NVMM zero-copy at all — the only
# JP5 entry point, NvBufSurfaceFromFd, is process-local.
#
# Probed via has_header_symbol so cross-builds against the right headers
# in Docker also work.
nvmm_target = 'unknown'
if have_nvbufsurface
  nvmm_hdr_dirs = []
  fsmod = import('fs')
  foreach d : ['/usr/src/jetson_multimedia_api/include',
               '/usr/include/nvbufsurface']
    if fsmod.is_dir(d)
      nvmm_hdr_dirs += '-I' + d
    endif
  endforeach
  have_import_api = cpp.has_header_symbol(
    'nvbufsurface.h', 'NvBufSurfaceImport',
    args : nvmm_hdr_dirs,
    required : false)
  if not have_import_api
    error('NvBufSurfaceImport not present in nvbufsurface.h. ' +
          'Cross-process NVMM IPC requires L4T R35.3.1 (JetPack 5.1.1) or newer ' +
          'on the JP5 line, or any JP6. Detected an older L4T (probably R35.2.1 / ' +
          'JP 5.0.x). Upgrade jetson-multimedia-api or use -Dmock=true for host tests.')
  endif
  nvmm_target = 'jetson (real NvBufSurface)'
else
  # Mock build — mock headers carry NvBufSurfaceImport stubs.
  nvmm_target = 'mock'
endif

message('NVMM IPC backend: pool + SCM_RIGHTS  (target: @0@)'.format(nvmm_target))

# NvSciBuf currently unused (no public headers ship with nvidia-l4t-nvsci);
# left here for the day NVIDIA exposes the API.
have_nvscibuf = false

# Configuration header
conf = configuration_data()
conf.set10('HAVE_NVBUFSURFACE', have_nvbufsurface)
conf.set10('HAVE_NVBUFSURFTRANSFORM', have_nvbufsurftransform)
conf.set10('HAVE_NVSCIBUF', have_nvscibuf)
conf.set_quoted('JETPACK_VERSION', nvmm_target)
conf.set_quoted('PACKAGE_VERSION', meson.project_version())

# Tunable defaults — see gst/common/nvmm_config.h for the full list.
conf.set('NVMM_POOL_RELEASE_DELAY', get_option('pool_release_delay'))
conf.set('NVMM_POLL_INTERVAL_US',   get_option('poll_interval_us'))

configure_file(
  output : 'config.h',
  configuration : conf,
)

config_inc = include_directories('.')
common_inc = include_directories('gst/common')

# Mock vs real NVMM: explicit opt-in via -Dmock=true. No silent fallback —
# if NvBufSurface isn't available and mock isn't requested, fail at configure.
# This catches the "I thought I was building for Jetson but actually shipped a
# CPU-only mock" mistake at build time, not at deploy time.
build_mock = get_option('mock')
if build_mock and have_nvbufsurface
  warning('-Dmock=true overrides the real NvBufSurface found on this host. ' +
          'You probably want to drop -Dmock for a production build.')
endif
if not build_mock and not have_nvbufsurface
  error('NvBufSurface (NVIDIA Multimedia API) not found and -Dmock=false. ' +
        'This plugin requires the Tegra NVMM stack. ' +
        'Install jetson-multimedia-api on a Jetson, or configure with -Dmock=true ' +
        'for host-side unit testing only.')
endif
if build_mock
  message('Building with MOCK NvBufSurface API. NOT FOR PRODUCTION.')
endif

subdir('gst/common')
subdir('gst/nvmmalloc')

# Pool + SCM_RIGHTS IPC backend (one backend, both JetPacks).
subdir('gst/ipc_pool')

subdir('gst/nvmmconvert')
subdir('gst/nvmmsink')
subdir('gst/nvmmappsrc')
subdir('tests')
subdir('benchmarks')
