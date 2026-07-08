# SAMURAI (SAM2.1) visual tracker (Phase B). Shares the TrtEngine wrapper with
# nvmminfer (compiled in, not linked). Gated on TRT + real NVMM (Jetson-only),
# same as nvmminfer/nvmmsecondaryinfer.
#
# VPI (when present) enables the GMC fft-cuda backend (gmc_vpi_fft.hpp); without
# it that backend compiles to a stub and the tracker falls back to fft-cpu/ncc.
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
  # CUDA std/arch from the shared samurai_cuda_base_args (build options
  # gpu_cxx_std / gpu_arch; default c++20 / sm_87 = Orin on CUDA 12.6). Strict:
  # warnings-as-errors for kernels + host-side -Wall/-Wextra via nvcc's host cc.
  cuda_args : samurai_cuda_base_args +
              ['-Werror', 'all-warnings', '-Xcompiler', '-Wall,-Wextra'],
  cpp_args : samurai_cpp_args,   # host samurai warnings-as-errors + NVMM_HAVE_VPI when present
  # Link with the C++ driver, not nvcc: for a mixed C++/CUDA target meson picks
  # the CUDA linker, which emits the toolkit -Wl,-rpath unwrapped and nvcc then
  # rejects ("Unknown option '-Wl,-rpath,...'"). The .cu.o objects link fine
  # under g++; cudart arrives via trt_dep and the kernels use no relocatable
  # device code, so no device-link step is dropped.
  link_language : 'cpp',
  install : true,
  install_dir : get_option('libdir') / 'gstreamer-1.0',
)
