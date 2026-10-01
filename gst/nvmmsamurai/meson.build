samurai_deps = [
  nvmm_common_dep, gstnvmmalloc_dep,
  gst_video_dep, gst_base_dep,
  trt_dep, nvbufsurftransform_dep,
]
samurai_cpp_args = ['-Werror']
if have_vpi
  samurai_deps += vpi_dep
  samurai_cpp_args += '-DNVMM_HAVE_VPI'
endif

gstnvmmsamurai = library('gstnvmmsamurai',
  files(
    'gstnvmmsamurai.cpp',
    'samurai_tracker.cpp',
    'samurai_kernels.cu',
    '../nvmminfer/trt_engine.cpp',
    '../nvmmsecondaryinfer/roi_preprocess.cpp',
  ),
  include_directories : [
    config_inc,
    common_inc,
    include_directories('../nvmmalloc'),
    include_directories('../nvmminfer'),
    include_directories('../nvmmsecondaryinfer'),
  ],
  dependencies : samurai_deps,
  cuda_args : samurai_cuda_base_args +
              ['-Werror', 'all-warnings', '-Xcompiler', '-Wall,-Wextra'],
  cpp_args : samurai_cpp_args,
  link_language : 'cpp',
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)
