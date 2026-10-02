tests_inc = include_directories('.')

mock_args = build_mock ? ['-DNVMM_MOCK_API'] : []

test_surface = executable('test_nvmm_surface',
  'test_nvmm_surface.cpp',
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep],
  cpp_args : mock_args,
)

test_gst_meta = executable('test_gst_meta_and_allocator',
  'test_gst_meta_and_allocator.cpp',
  include_directories : [config_inc, common_inc, include_directories('../gst/nvmmalloc')],
  dependencies : [nvmm_common_dep, gstnvmmalloc_dep, gst_dep],
  cpp_args : mock_args,
)

test_host_cores = executable('test_host_cores',
  'test_host_cores.cpp',
  '../gst/nvmminfer/yolo_parser.cpp',
  '../gst/nvmmtracker/tracker.cpp',
  '../gst/common/nvmm_motion.cpp',
  '../gst/nvmmsecondaryinfer/secondary_cache.cpp',
  include_directories : [config_inc, common_inc,
                         include_directories('../gst/nvmminfer'),
                         include_directories('../gst/nvmmtracker'),
                         include_directories('../gst/nvmmsecondaryinfer')],
  cpp_args : mock_args,
)

test_kalman_box = executable('test_kalman_box',
  ['test_kalman_box.cpp', '../gst/common/kalman_box.cpp'],
  include_directories : [common_inc],
)

# -Wconversion and -Wdouble-promotion only flag intentional double paths here.
gmc_test_args = ['-Wall', '-Wextra', '-Werror', '-Wshadow']

test_pure_cpp = executable('test_pure_cpp',
  ['test_pure_cpp.cpp'],
  include_directories : [common_inc, include_directories('../gst/nvmmsamurai')],
  cpp_args : gmc_test_args,
)

# Plain build is the ASan+UBSan sweep; scripts/run-fuzz.sh builds the libFuzzer one.
test_fuzz_gmc = executable('test_fuzz_gmc',
  ['fuzz_gmc.cpp'],
  include_directories : [common_inc, include_directories('../gst/nvmmsamurai')],
  cpp_args : gmc_test_args,
)

if have_tensorrt and have_nvbufsurface
  samurai_kernel_probe = executable('samurai_kernel_probe',
    ['samurai_kernel_probe.cpp', '../gst/nvmmsamurai/samurai_kernels.cu'],
    include_directories : [include_directories('../gst/nvmmsamurai')],
    dependencies : [trt_dep],
    cuda_args : samurai_cuda_base_args,
    # nvcc rejects the unwrapped -Wl,-rpath meson emits for mixed C++/CUDA links.
    link_language : 'cpp',
  )
  # cudaMalloc OOMs under TSan's shadow reservation.
  test('samurai_kernels', samurai_kernel_probe, protocol : 'exitcode',
    suite : 'nvidia_hwlib')
endif

# On real NVMM, libnvbufsurftransform double-locks its own mutex and cudaMalloc
# OOMs under TSan; the on-device TSan run skips nvidia_hwlib.
hwlib_suite = build_mock ? [] : ['nvidia_hwlib']
test('nvmm_surface', test_surface, protocol : 'exitcode', suite : hwlib_suite)
test('gst_meta_and_allocator', test_gst_meta, protocol : 'exitcode')
test('host_cores', test_host_cores, protocol : 'exitcode')
# No GLib, so run-sanitizers.sh can run this suite with detect_leaks=1.
test('kalman_box', test_kalman_box, protocol : 'exitcode', suite : 'pure_cpp')
test('pure_cpp_headers', test_pure_cpp, protocol : 'exitcode', suite : 'pure_cpp')
# 'fuzz' because the mutation gate skips it per mutant: most of the suite's wall time.
test('fuzz_gmc', test_fuzz_gmc, protocol : 'exitcode', timeout : 300,
  suite : ['pure_cpp', 'fuzz'])

shm_element_inc = [
  config_inc,
  common_inc,
  include_directories('../gst/nvmmalloc'),
  include_directories('../gst/nvmmsink'),
  include_directories('../gst/nvmmappsrc'),
]

plugin_path = meson.current_build_dir() / '..' / 'gst' / 'nvmmsink' + ':' + \
              meson.current_build_dir() / '..' / 'gst' / 'nvmmappsrc' + ':' + \
              meson.current_build_dir() / '..' / 'gst' / 'nvmmalloc'

plugin_path_with_convert = plugin_path + ':' + \
    meson.current_build_dir() / '..' / 'gst' / 'nvmmconvert'

compositor_plugin_path = meson.current_build_dir() / '..' / 'gst' / 'nvmmcompositor' \
    + ':' + meson.current_build_dir() / '..' / 'gst' / 'nvmmalloc'

test_compositor = executable('test_nvmm_compositor',
  'test_nvmm_compositor.cpp',
  include_directories : [config_inc, common_inc],
  dependencies : [gst_dep, gst_base_dep],
  cpp_args : mock_args,
)

test_sink_appsrc = executable('test_nvmm_sink_appsrc',
  'test_nvmm_sink_appsrc.cpp',
  include_directories : shm_element_inc,
  dependencies : [gstnvmmsink_dep, gstnvmmappsrc_dep, gst_dep],
  cpp_args : mock_args,
)

test_integration = executable('test_integration',
  'test_integration.cpp',
  include_directories : shm_element_inc,
  dependencies : [gstnvmmsink_dep, gstnvmmappsrc_dep, gst_dep, gst_video_dep],
  cpp_args : mock_args,
)

# A TSan-built plugin .so cannot be loaded by the unsanitized plugin scanner.
test('nvmm_compositor', test_compositor, protocol : 'exitcode',
  is_parallel : false, suite : 'plugin',
  env : ['GST_PLUGIN_PATH=' + compositor_plugin_path])
test('nvmm_sink_appsrc', test_sink_appsrc, protocol : 'exitcode',
  is_parallel : false, suite : 'plugin',
  env : ['GST_PLUGIN_PATH=' + plugin_path])
test('integration', test_integration, protocol : 'exitcode',
  is_parallel : false, suite : 'plugin',
  env : ['GST_PLUGIN_PATH=' + plugin_path_with_convert])

if have_analytics
  test_analytics = executable('test_analytics', 'test_analytics.cpp',
    include_directories : [analytics_inc],
  )
  test('analytics', test_analytics, protocol : 'exitcode')
endif

if have_analytics_golden
  foreach t : ['image_ops', 'low_texture_motion', 'active_region', 'motion_magnify',
               'dual_homography']
    golden_exe = executable('test_golden_' + t, 'test_golden_' + t + '.cpp',
      include_directories : [analytics_inc],
      dependencies : [opencv4_dep],
    )
    test('golden_' + t, golden_exe, protocol : 'exitcode', timeout : 120)
  endforeach
endif

if have_analytics_cuda
  analytics_kernel_probe = executable('analytics_kernel_probe',
    ['analytics_kernel_probe.cpp', '../analytics/analytics_kernels.cu'],
    include_directories : [analytics_inc],
    dependencies : [analytics_cuda_dep],
    cuda_args : analytics_cuda_args,
    # nvcc rejects the unwrapped -Wl,-rpath meson emits for mixed C++/CUDA links.
    link_language : 'cpp',
  )
  test('analytics_kernels', analytics_kernel_probe, protocol : 'exitcode',
    suite : 'nvidia_hwlib')
endif

element_test_inc = [config_inc, common_inc, include_directories('../gst/nvmmalloc')]
gst_build_dir = meson.current_build_dir() / '..' / 'gst'
gstcheck_dep = dependency('gstreamer-check-1.0',
  required : (have_vpi or have_tensorrt) and have_nvbufsurface)

if have_vpi and have_nvbufsurface
  test_ofa = executable('test_nvmm_ofa', 'test_nvmm_ofa.cpp',
    include_directories : element_test_inc,
    dependencies : [gstnvmmalloc_dep, gstcheck_dep, gst_video_dep, nvbufsurftransform_dep,
                    vpi_dep],
  )
  test('nvmm_ofa', test_ofa, protocol : 'exitcode', is_parallel : false,
    suite : ['plugin', 'nvidia_hwlib'],
    # VPI OFA segfaults under perturbed malloc with no project code involved (#118).
    env : ['MALLOC_PERTURB_=0',
           'GST_PLUGIN_PATH=' + gst_build_dir / 'nvmmofa' + ':' + gst_build_dir / 'nvmmalloc'])
endif

if have_tensorrt and have_nvbufsurface
  element_test_deps = [gstnvmmalloc_dep, gstcheck_dep, gst_video_dep, trt_dep,
                       nvbufsurftransform_dep]

  test_drawdet = executable('test_nvmm_drawdet', 'test_nvmm_drawdet.cpp',
    include_directories : element_test_inc,
    dependencies : element_test_deps,
  )
  test('nvmm_drawdet', test_drawdet, protocol : 'exitcode', is_parallel : false,
    suite : ['plugin', 'nvidia_hwlib'],
    env : ['GST_PLUGIN_PATH=' + gst_build_dir / 'nvmmdrawdet' + ':' + gst_build_dir / 'nvmmalloc'])

  test_preprocess = executable('test_nvmm_preprocess',
    ['test_nvmm_preprocess.cpp', '../gst/nvmminfer/preprocess.cpp',
     '../gst/nvmmsecondaryinfer/roi_preprocess.cpp'],
    include_directories : element_test_inc + [include_directories('../gst/nvmminfer'),
                                               include_directories('../gst/nvmmsecondaryinfer')],
    dependencies : element_test_deps,
  )
  test('nvmm_preprocess', test_preprocess, protocol : 'exitcode', suite : 'nvidia_hwlib')

  test_trt_elements = executable('test_nvmm_trt_elements', 'test_nvmm_trt_elements.cpp',
    include_directories : element_test_inc,
    dependencies : element_test_deps,
  )
  trt_plugin_path = gst_build_dir / 'nvmmalloc'
  foreach p : ['nvmminfer', 'nvmmsecondaryinfer', 'nvmmdetgate', 'nvmmsamurai']
    trt_plugin_path += ':' + gst_build_dir / p
  endforeach
  test('nvmm_trt_elements', test_trt_elements, protocol : 'exitcode', is_parallel : false,
    suite : ['plugin', 'nvidia_hwlib'], timeout : 120,
    env : ['GST_PLUGIN_PATH=' + trt_plugin_path])
endif

if have_tensorrt and have_nvbufsurface
  engine_backed_test_deps = [gstnvmmalloc_dep, gstcheck_dep, gst_video_dep, nvbufsurftransform_dep]
  foreach t : [['detgate', 'nvmmdetgate'], ['samurai', 'nvmmsamurai']]
    exe = executable('test_nvmm_' + t[0], 'test_nvmm_' + t[0] + '.cpp',
      include_directories : element_test_inc,
      dependencies : engine_backed_test_deps,
    )
    test('nvmm_' + t[0], exe, protocol : 'exitcode', is_parallel : false,
      suite : ['plugin', 'nvidia_hwlib'], timeout : 180,
      env : ['GST_PLUGIN_PATH=' + gst_build_dir / t[1] + ':' + gst_build_dir / 'nvmmalloc'])
  endforeach
endif
