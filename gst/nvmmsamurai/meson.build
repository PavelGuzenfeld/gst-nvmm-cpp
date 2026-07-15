# SAMURAI (SAM2.1) visual tracker (Phase B). Shares the TrtEngine wrapper with
# nvmminfer (compiled in, not linked). Gated on TRT + real NVMM (Jetson-only),
# same as nvmminfer/nvmmsecondaryinfer.
gstnvmmsamurai = library('gstnvmmsamurai',
  files(
    'gstnvmmsamurai.cpp',
    'samurai_tracker.cpp',
    'samurai_kernels.cu',
    '../nvmminfer/trt_engine.cpp',
    '../nvmmsecondaryinfer/roi_preprocess.cpp',
    '../common/xfeat_matcher.cpp',   # XFeat+LightGlue matcher (GMC + validity)
  ),
  include_directories : [
    config_inc,
    common_inc,
    include_directories('../nvmmalloc'),
    include_directories('../nvmminfer'),
    include_directories('../nvmmsecondaryinfer'),
  ],
  dependencies : [
    nvmm_common_dep, gstnvmmalloc_dep,
    gst_video_dep, gst_base_dep,
    trt_dep, nvbufsurftransform_dep,
  ],
  # CUDA std/arch from the shared samurai_cuda_base_args (build options
  # gpu_cxx_std / gpu_arch; default c++20 / sm_87 = Orin on CUDA 12.6). Strict:
  # warnings-as-errors for kernels + host-side -Wall/-Wextra via nvcc's host cc.
  cuda_args : samurai_cuda_base_args +
              ['-Werror', 'all-warnings', '-Xcompiler', '-Wall,-Wextra'],
  cpp_args : ['-Werror'],   # host samurai code warnings-as-errors (already -Wall/-Wextra via level 2)
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)
