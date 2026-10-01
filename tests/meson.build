tests_inc = include_directories('.')

test_buffer = executable('test_nvmm_buffer',
  'test_nvmm_buffer.cpp',
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test_transform = executable('test_nvmm_transform',
  'test_nvmm_transform.cpp',
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test_gst_allocator = executable('test_gst_nvmm_allocator',
  'test_gst_nvmm_allocator.cpp',
  include_directories : [
    config_inc,
    common_inc,
    include_directories('../gst/nvmmalloc'),
  ],
  dependencies : [nvmm_common_dep, gstnvmmalloc_dep, gst_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test_fuzz_shm = executable('fuzz_shm_header',
  'fuzz_shm_header.cpp',
  include_directories : [config_inc, common_inc],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test_ofa_meta = executable('test_optical_flow_meta',
  'test_optical_flow_meta.cpp',
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test_yolo_parser = executable('test_yolo_parser',
  'test_yolo_parser.cpp',
  '../gst/nvmminfer/yolo_parser.cpp',
  include_directories : [config_inc, common_inc, include_directories('../gst/nvmminfer')],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test_nvmm_motion = executable('test_nvmm_motion',
  'test_nvmm_motion.cpp',
  '../gst/common/nvmm_motion.cpp',
  include_directories : [config_inc, common_inc],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test_nvmm_tracker = executable('test_nvmm_tracker',
  'test_nvmm_tracker.cpp',
  '../gst/nvmmtracker/tracker.cpp',
  include_directories : [config_inc, common_inc, include_directories('../gst/nvmmtracker')],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test_secondary_cache = executable('test_secondary_cache',
  'test_secondary_cache.cpp',
  '../gst/nvmmsecondaryinfer/secondary_cache.cpp',
  include_directories : [config_inc, common_inc,
                         include_directories('../gst/nvmmsecondaryinfer')],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test_class_meta = executable('test_nvmm_class_meta',
  'test_nvmm_class_meta.cpp',
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep, gst_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

rt_dep = cpp.find_library('rt', required : true)

test_sink = executable('test_nvmm_sink',
  'test_nvmm_sink.cpp',
  include_directories : [
    config_inc,
    common_inc,
    include_directories('../gst/nvmmalloc'),
    include_directories('../gst/nvmmsink'),
  ],
  dependencies : [gstnvmmsink_dep, gst_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test_det_meta = executable('test_nvmm_det_meta',
  'test_nvmm_det_meta.cpp',
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep, gst_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test_track_meta = executable('test_nvmm_track_meta',
  'test_nvmm_track_meta.cpp',
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep, gst_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test_kalman_box = executable('test_kalman_box',
  ['test_kalman_box.cpp', '../gst/common/kalman_box.cpp'],
  include_directories : [common_inc],
)

test_xfeat_motion = executable('test_xfeat_motion',
  ['test_xfeat_motion.cpp'],
  include_directories : [common_inc],
)

test_phase_correlation = executable('test_phase_correlation',
  'test_phase_correlation.cpp',
  include_directories : [common_inc],
)

test_samurai_view = executable('test_samurai_view',
  ['test_samurai_view.cpp'],
  include_directories : [include_directories('../gst/nvmmsamurai')],
)

# -Wconversion and -Wdouble-promotion only flag intentional double paths here.
gmc_test_args = ['-Wall', '-Wextra', '-Werror', '-Wshadow']

test_gmc_backend = executable('test_gmc_backend',
  ['test_gmc_backend.cpp'],
  include_directories : [common_inc, include_directories('../gst/nvmmsamurai')],
  cpp_args : gmc_test_args,
)

# Plain build is the ASan+UBSan sweep; scripts/run-fuzz.sh builds the libFuzzer one.
test_fuzz_gmc = executable('test_fuzz_gmc',
  ['fuzz_gmc.cpp'],
  include_directories : [common_inc, include_directories('../gst/nvmmsamurai')],
  cpp_args : gmc_test_args,
)

test_gmc_mask = executable('test_gmc_mask',
  ['test_gmc_mask.cpp'],
  include_directories : [include_directories('../gst/nvmmsamurai')],
  cpp_args : gmc_test_args,
)

test_vit_grid = executable('test_vit_grid',
  ['test_vit_grid.cpp'],
  include_directories : [common_inc],
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
test('nvmm_buffer', test_buffer, protocol : 'exitcode', suite : hwlib_suite)
test('nvmm_transform', test_transform, protocol : 'exitcode', suite : hwlib_suite)
test('gst_nvmm_allocator', test_gst_allocator, protocol : 'exitcode')
test('fuzz_shm_header', test_fuzz_shm, protocol : 'exitcode', args : ['200000'],
  suite : 'fuzz')
test('optical_flow_meta', test_ofa_meta, protocol : 'exitcode')
test('nvmm_det_meta', test_det_meta, protocol : 'exitcode')
test('nvmm_track_meta', test_track_meta, protocol : 'exitcode')
# No GLib, so run-sanitizers.sh can run this suite with detect_leaks=1.
test('kalman_box', test_kalman_box, protocol : 'exitcode', suite : 'pure_cpp')
test('phase_correlation', test_phase_correlation, protocol : 'exitcode', suite : 'pure_cpp')
test('xfeat_motion', test_xfeat_motion, protocol : 'exitcode', suite : 'pure_cpp')
test('gmc_backend', test_gmc_backend, protocol : 'exitcode', suite : 'pure_cpp')
# 'fuzz' because the mutation gate skips it per mutant: most of the suite's wall time.
test('fuzz_gmc', test_fuzz_gmc, protocol : 'exitcode', timeout : 300,
  suite : ['pure_cpp', 'fuzz'])
test('gmc_mask', test_gmc_mask, protocol : 'exitcode', suite : 'pure_cpp')
test('samurai_view', test_samurai_view, protocol : 'exitcode', suite : 'pure_cpp')
test('vit_grid', test_vit_grid, protocol : 'exitcode', suite : 'pure_cpp')
test('yolo_parser', test_yolo_parser, protocol : 'exitcode')
test('nvmm_tracker', test_nvmm_tracker, protocol : 'exitcode')
test('nvmm_motion', test_nvmm_motion, protocol : 'exitcode')
test('secondary_cache', test_secondary_cache, protocol : 'exitcode')
test('nvmm_class_meta', test_class_meta, protocol : 'exitcode')
test_appsrc = executable('test_nvmm_appsrc',
  'test_nvmm_appsrc.cpp',
  include_directories : [
    config_inc,
    common_inc,
    include_directories('../gst/nvmmalloc'),
    include_directories('../gst/nvmmsink'),
    include_directories('../gst/nvmmappsrc'),
  ],
  dependencies : [gstnvmmsink_dep, gstnvmmappsrc_dep, gst_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

plugin_path = meson.current_build_dir() / '..' / 'gst' / 'nvmmsink' + ':' + \
              meson.current_build_dir() / '..' / 'gst' / 'nvmmappsrc' + ':' + \
              meson.current_build_dir() / '..' / 'gst' / 'nvmmalloc'

test_gstcheck = executable('test_gstcheck_elements',
  'test_gstcheck_elements.cpp',
  include_directories : [config_inc, common_inc],
  dependencies : [gst_dep, gst_video_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

plugin_path_with_convert = plugin_path + ':' + \
    meson.current_build_dir() / '..' / 'gst' / 'nvmmconvert'

compositor_plugin_path = meson.current_build_dir() / '..' / 'gst' / 'nvmmcompositor' \
    + ':' + meson.current_build_dir() / '..' / 'gst' / 'nvmmalloc'

test_compositor = executable('test_nvmm_compositor',
  'test_nvmm_compositor.cpp',
  include_directories : [config_inc, common_inc],
  dependencies : [gst_dep, gst_base_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

# A TSan-built plugin .so cannot be loaded by the unsanitized plugin scanner.
test('nvmm_compositor', test_compositor, protocol : 'exitcode',
  is_parallel : false, suite : 'plugin',
  env : ['GST_PLUGIN_PATH=' + compositor_plugin_path])

test('nvmm_sink', test_sink, protocol : 'exitcode',
  is_parallel : false, suite : 'plugin',
  env : ['GST_PLUGIN_PATH=' + plugin_path])
test('nvmm_appsrc', test_appsrc, protocol : 'exitcode',
  is_parallel : false, suite : 'plugin',
  env : ['GST_PLUGIN_PATH=' + plugin_path])
test('gstcheck_elements', test_gstcheck, protocol : 'exitcode',
  is_parallel : false, suite : 'plugin',
  env : ['GST_PLUGIN_PATH=' + plugin_path_with_convert])

test_integration = executable('test_integration',
  'test_integration.cpp',
  include_directories : [
    config_inc,
    common_inc,
    include_directories('../gst/nvmmalloc'),
    include_directories('../gst/nvmmsink'),
    include_directories('../gst/nvmmappsrc'),
  ],
  dependencies : [gstnvmmsink_dep, gstnvmmappsrc_dep, gst_dep, gst_video_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test('integration', test_integration, protocol : 'exitcode',
  is_parallel : false, suite : 'plugin',
  env : ['GST_PLUGIN_PATH=' + plugin_path_with_convert])

if have_analytics
  foreach t : ['persistence_gate', 'dual_homography', 'low_texture_motion',
               'active_region', 'detection_motion_gate', 'motion_magnify']
    analytics_exe = executable('test_analytics_' + t, 'test_analytics_' + t + '.cpp',
      include_directories : [analytics_inc],
    )
    test('analytics_' + t, analytics_exe, protocol : 'exitcode')
  endforeach
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
