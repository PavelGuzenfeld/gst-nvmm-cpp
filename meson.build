project('gst-nvmm-cpp', ['c', 'cpp'],
  version : '0.1.0',
  meson_version : '>= 0.62.0',
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

nvbufsurface_dep = dependency('nvbufsurface', required : false)
if not nvbufsurface_dep.found()
  # Fallback: search standard Jetson paths
  nvbufsurface_lib = cc.find_library('nvbufsurface',
    dirs : ['/usr/lib/aarch64-linux-gnu/tegra', '/usr/lib/aarch64-linux-gnu'],
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
    dirs : ['/usr/lib/aarch64-linux-gnu/tegra', '/usr/lib/aarch64-linux-gnu'],
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

# JetPack version detection via L4T version file
jetpack_version = 'unknown'
l4t_release = run_command('bash', '-c',
  'cat /etc/nv_tegra_release 2>/dev/null | head -1 | sed "s/.*R\\([0-9]*\\).*/\\1/"',
  check : false,
)
if l4t_release.returncode() == 0
  l4t_major = l4t_release.stdout().strip()
  if l4t_major == '35'
    jetpack_version = '5'
  elif l4t_major == '36'
    jetpack_version = '6'
  endif
endif

# NvSciBuf support (JP6+ only)
have_nvscibuf = false
if jetpack_version == '6'
  nvscibuf_dep = dependency('nvscibuf', required : false)
  if not nvscibuf_dep.found()
    nvscibuf_lib = cc.find_library('nvscibuf', required : false)
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
subdir('gst/nvmmalloc')
subdir('gst/nvmmconvert')
subdir('gst/nvmmsink')
subdir('gst/nvmmappsrc')
subdir('tests')
subdir('benchmarks')
