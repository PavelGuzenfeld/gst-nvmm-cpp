project('gst-nvmm-cpp', ['c', 'cpp'],
  version : files('VERSION'),  # single source of truth — bump VERSION only
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
    nvscibuf_lib = cc.find_library('nvscibuf', dirs : nvmm_lib_dirs, required : false)
    if nvscibuf_lib.found()
      nvscibuf_dep = declare_dependency(dependencies : nvscibuf_lib)
      have_nvscibuf = true
    endif
  else
    have_nvscibuf = true
  endif
endif

# VPI support — needed for the OFA optical-flow element (nvmmofa). VPI ships
# under /opt/nvidia/vpi{2,3} with no pkg-config, so probe the lib + headers.
# Note: OFA is Orin-only hardware; the element builds wherever VPI is present
# and gates OFA availability at runtime.
have_vpi = false
fs = import('fs')
vpi_lib_dirs = [
  '/opt/nvidia/vpi3/lib/aarch64-linux-gnu',
  '/opt/nvidia/vpi2/lib/aarch64-linux-gnu',
]
vpi_inc_candidates = ['/opt/nvidia/vpi3/include', '/opt/nvidia/vpi2/include']
nvvpi_lib = cc.find_library('nvvpi', dirs : vpi_lib_dirs, required : false)
if nvvpi_lib.found()
  vpi_inc_dirs = []
  foreach d : vpi_inc_candidates
    if fs.is_dir(d)
      vpi_inc_dirs += d
    endif
  endforeach
  if vpi_inc_dirs.length() > 0
    vpi_dep = declare_dependency(
      dependencies : nvvpi_lib,
      include_directories : include_directories(vpi_inc_dirs, is_system : true),
    )
    have_vpi = true
  endif
endif

# TensorRT + CUDA + NPP — needed for the inference element (nvmminfer). These ship
# under /usr/lib/aarch64-linux-gnu + /usr/local/cuda with no pkg-config on Jetson,
# so probe the libs + headers directly. The element builds wherever TensorRT is
# present (Jetson) and is gated out of the x86 mock/CI build (skip-on-host, the
# same strategy as VPI/nvmmofa; CUDA/TRT are not mocked).
have_tensorrt = false
trt_header_dir = ''
foreach d : ['/usr/include/aarch64-linux-gnu', '/usr/include']
  if trt_header_dir == '' and fs.is_file(d / 'NvInfer.h')
    trt_header_dir = d
  endif
endforeach
cuda_lib_dirs = ['/usr/local/cuda/lib64', '/usr/lib/aarch64-linux-gnu']
nvinfer_lib = cc.find_library('nvinfer', dirs : cuda_lib_dirs, required : false)
cudart_lib  = cc.find_library('cudart',  dirs : cuda_lib_dirs, required : false)
nppc_lib    = cc.find_library('nppc',    dirs : cuda_lib_dirs, required : false)
nppidei_lib = cc.find_library('nppidei', dirs : cuda_lib_dirs, required : false)
nppig_lib   = cc.find_library('nppig',   dirs : cuda_lib_dirs, required : false)
nppial_lib  = cc.find_library('nppial',  dirs : cuda_lib_dirs, required : false)
if trt_header_dir != '' and fs.is_dir('/usr/local/cuda/include') \
   and nvinfer_lib.found() and cudart_lib.found() \
   and nppc_lib.found() and nppidei_lib.found() and nppig_lib.found() and nppial_lib.found()
  trt_dep = declare_dependency(
    dependencies : [nvinfer_lib, cudart_lib, nppc_lib, nppidei_lib, nppig_lib, nppial_lib],
    include_directories : include_directories(
      trt_header_dir, '/usr/local/cuda/include', is_system : true),
  )
  have_tensorrt = true
endif

# Optional DeepStream metadata bridge (libnvds_meta + nvdsmeta.h). Off by default
# so the suite stays DeepStream-free; the flat IPC wire format never needs it.
have_deepstream_meta = false
nvds_meta_dep = declare_dependency()  # empty placeholder; only consumed when enabled below
if get_option('enable_deepstream_meta')
  # NvDsBatchMeta pool/iterators live in libnvds_meta; gst_buffer_get_nvds_batch_meta
  # lives in libnvdsgst_meta. The producer-side extraction needs both.
  nvds_meta_lib = cc.find_library('nvds_meta', dirs : nvmm_lib_dirs, required : true)
  nvdsgst_meta_lib = cc.find_library('nvdsgst_meta', dirs : nvmm_lib_dirs, required : true)
  fsmod_ds = import('fs')
  nvds_inc_candidates = [
    '/opt/nvidia/deepstream/deepstream/sources/includes',
  ]
  nvds_inc_dirs = []
  foreach d : nvds_inc_candidates
    if fsmod_ds.is_dir(d)
      nvds_inc_dirs += d
    endif
  endforeach
  if nvds_inc_dirs.length() == 0
    error('enable_deepstream_meta=true but DeepStream includes (nvdsmeta.h) not found')
  endif
  nvds_meta_dep = declare_dependency(
    dependencies : [nvds_meta_lib, nvdsgst_meta_lib],
    include_directories : include_directories(nvds_inc_dirs, is_system : true),
    compile_args : ['-DNVMM_DEEPSTREAM_META'],
  )
  have_deepstream_meta = true
endif

# Configuration header
conf = configuration_data()
conf.set10('HAVE_NVBUFSURFACE', have_nvbufsurface)
conf.set10('HAVE_NVBUFSURFTRANSFORM', have_nvbufsurftransform)
conf.set10('HAVE_NVSCIBUF', have_nvscibuf)
conf.set10('HAVE_VPI', have_vpi)
conf.set10('HAVE_TENSORRT', have_tensorrt)
conf.set10('HAVE_DEEPSTREAM_META', have_deepstream_meta)
conf.set_quoted('JETPACK_VERSION', jetpack_version)
conf.set_quoted('PACKAGE_VERSION', meson.project_version())

configure_file(
  output : 'config.h',
  configuration : conf,
)

config_inc = include_directories('.')
common_inc = include_directories('gst/common')

# Optional analytics feature (OFF by default). The header-only motion-analytics
# components (analytics/) are always exposed via analytics_inc and are pure C++14
# — no OpenCV. The feature only gates their unit tests, so it needs no external
# dependency at all.
have_analytics = get_option('analytics').allowed()
if have_analytics
  message('analytics feature enabled (building analytics unit tests)')
endif

# Golden-comparison lane (OFF by default): builds the tests + benchmarks that
# validate the hand-rolled analytics against OpenCV as a reference oracle. The
# ONLY thing in the tree that needs OpenCV. Enabled => OpenCV is required (no
# silent half-state).
opencv4_dep = dependency('opencv4', required : get_option('analytics_golden'),
  include_type : 'system')
have_analytics_golden = opencv4_dep.found()
if have_analytics_golden
  message('analytics_golden enabled (OpenCV @0@ as reference oracle)'.format(opencv4_dep.version()))
endif

# CUDA analytics lane (OFF by default). A narrow nvcc+cudart gate — analytics
# kernels need neither TensorRT nor NvBufSurface, so this is deliberately NOT
# the Jetson-only have_tensorrt gate. Strict opt-in (enabled(), not allowed()):
# requiring nvcc under 'auto' would break every non-CUDA host.
have_analytics_cuda = false
if get_option('analytics_cuda').enabled()
  add_languages('cuda', native : false, required : true)
  analytics_cuda_dep = dependency('cuda', required : true, modules : ['cudart'])
  analytics_cuda_args = ['-std=' + get_option('gpu_cxx_std'),
                         '--gpu-architecture=' + get_option('gpu_arch')]
  have_analytics_cuda = true
  message('analytics_cuda enabled (@0@, @1@)'.format(get_option('gpu_arch'),
                                                     get_option('gpu_cxx_std')))
endif

# When NVMM libs not available, build mock versions for testing
build_mock = not have_nvbufsurface
if build_mock
  message('NvBufSurface not found — building with mock API for host-side testing')
endif

subdir('gst/common')
subdir('analytics')        # header-only motion-analytics components (pure C++14, no OpenCV)
subdir('gst/nvmmalloc')
subdir('gst/nvmmconvert')
subdir('gst/nvmmcompositor')
subdir('gst/nvmmsink')
subdir('gst/nvmmappsrc')
subdir('gst/nvmmtracker')  # pure-host det-meta tracker (no CUDA), builds everywhere
subdir('gst/nvmmfusion')   # pure-host PTS branch fusion (no CUDA), builds everywhere
subdir('gst/nvmmfusekf')   # pure-host master-KF YOLO+SAMURAI fusion (no CUDA)
subdir('gst/nvmmexample')  # docs/extending.md worked example (built, never installed)
# OFA optical flow (nvmmofa) + its flow-meta consumer — needs VPI + real NVMM
# (Orin-only hardware; gated out of the x86 mock/CI build).
if have_vpi and have_nvbufsurface
  message('VPI found — building nvmmofa (OFA optical flow)')
  subdir('gst/nvmmofa')
endif
# TensorRT inference (nvmminfer) — needs TRT + CUDA + NPP + real NVMM (Jetson-only;
# gated out of the x86 mock/CI build, same as nvmmofa).
if have_tensorrt and have_nvbufsurface
  message('TensorRT found — building nvmminfer (TRT inference)')
  # CUDA language for the SAMURAI per-frame kernels (Jetson-only; x86 CI has no
  # nvcc and doesn't build these elements). std/arch come from the cuda_std /
  # cuda_arch build options (default c++20 / sm_87 = Orin on CUDA 12.6) and are
  # passed per-target via cuda_args (nvmmsamurai + tests/meson.build).
  add_languages('cuda', native : false, required : true)
  # Shared CUDA std/arch flags for every .cu target (nvmmsamurai + the samurai
  # test probes) — single source of truth so the probes can't compile against a
  # different arch/std than the shipped library. Targets append their own flags.
  samurai_cuda_base_args = ['-std=' + get_option('gpu_cxx_std'),
                            '--gpu-architecture=' + get_option('gpu_arch')]
  subdir('gst/nvmminfer')
  subdir('gst/nvmmsecondaryinfer')  # Phase-3.2 cascade classifier (TRT)
  subdir('gst/nvmmdrawdet')  # detection overlay (demo/viz/E2E), uses CUDA+VIC
  subdir('gst/nvmmsamurai')  # Phase-B SAM2.1 SAMURAI visual tracker (TRT)
endif
subdir('tests')
subdir('benchmarks')
subdir('examples')     # programmatic-pipeline worked example (core gst, CI-run, never installed)
