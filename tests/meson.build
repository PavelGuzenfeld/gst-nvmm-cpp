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

# Pure-host YOLO parser test: compiles the parser directly (no CUDA/TRT/gst), so
# the decode/NMS/letterbox-unmap math is covered on the x86 CI build, not only
# on a Jetson during manual E2E.
test_yolo_parser = executable('test_yolo_parser',
  'test_yolo_parser.cpp',
  '../gst/nvmminfer/yolo_parser.cpp',
  include_directories : [config_inc, common_inc, include_directories('../gst/nvmminfer')],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

# Pure-host motion-compute test: per-box mean flow + moving flag (Phase 3).
test_nvmm_motion = executable('test_nvmm_motion',
  'test_nvmm_motion.cpp',
  '../gst/common/nvmm_motion.cpp',
  include_directories : [config_inc, common_inc],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

# Pure-host tracker test: compiles the tracker core directly (no CUDA/TRT/gst),
# covering IOU matching + id persistence + expiry on the x86 CI build.
test_nvmm_tracker = executable('test_nvmm_tracker',
  'test_nvmm_tracker.cpp',
  '../gst/nvmmtracker/tracker.cpp',
  include_directories : [config_inc, common_inc, include_directories('../gst/nvmmtracker')],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

# Pure-host secondary-cache test: compiles the cascade's interval/expiry policy
# core directly (no CUDA/TRT/gst), covered on the x86 CI build (Phase 3.2).
test_secondary_cache = executable('test_secondary_cache',
  'test_secondary_cache.cpp',
  '../gst/nvmmsecondaryinfer/secondary_cache.cpp',
  include_directories : [config_inc, common_inc,
                         include_directories('../gst/nvmmsecondaryinfer')],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

# Classifier-result sibling meta (Phase 3.2): pure GstMeta round-trips.
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

# Pure-C++ Kalman filter (no gst/CUDA) — compile the unit under test directly.
test_kalman_box = executable('test_kalman_box',
  ['test_kalman_box.cpp', '../gst/common/kalman_box.cpp'],
  include_directories : [common_inc],
)

# Header-only crop geometry (gst/nvmmsamurai/samurai_view.hpp); no CUDA/GST.
test_samurai_view = executable('test_samurai_view',
  ['test_samurai_view.cpp'],
  include_directories : [include_directories('../gst/nvmmsamurai')],
)

# Header-only ViT grid-token count (gst/common/vit_grid.hpp); no CUDA/GST.
test_vit_grid = executable('test_vit_grid',
  ['test_vit_grid.cpp'],
  include_directories : [common_inc],
)

# Parity-checks the SAMURAI CUDA kernels (transpose, add_per_channel, sigmoid/
# threshold scale, bilinear upsample, mask bbox, memory assembly) against their
# host references using deterministic synthetic inputs — no engines or golden
# data, so it is a self-contained correctness test. CUDA-only, hence inside the
# have_tensorrt block; the mock/x86 CI build skips it.
if have_tensorrt and have_nvbufsurface
  samurai_kernel_probe = executable('samurai_kernel_probe',
    ['samurai_kernel_probe.cpp', '../gst/nvmmsamurai/samurai_kernels.cu'],
    include_directories : [include_directories('../gst/nvmmsamurai')],
    dependencies : [trt_dep],   # brings the CUDA include dir + cudart
    cuda_args : samurai_cuda_base_args,
    # Link via g++, not nvcc: meson's CUDA linker for a mixed C++/CUDA target
    # emits -Wl,-rpath unwrapped and nvcc rejects it. cudart arrives via trt_dep;
    # the kernels use no relocatable device code.
    link_language : 'cpp',
  )
  # Raw cudaMalloc — the same allocator that OOMs under TSan's shadow space, so
  # tag 'nvidia_hwlib' (the on-device TSan run skips that suite). This is a
  # real-GPU test; it only exists on the Jetson build.
  test('samurai_kernels', samurai_kernel_probe, protocol : 'exitcode',
    suite : 'nvidia_hwlib')
endif

# On a real-API (non-mock) build these two exercise closed NVIDIA libs that
# ThreadSanitizer flags but we cannot fix: libnvbufsurftransform double-locks /
# unlocks its own global mutex, and the CUDA allocator OOMs under TSan's shadow
# reservation. Tag them 'nvidia_hwlib' so the on-device TSan run skips them
# (run-sanitizers.sh passes --no-suite nvidia_hwlib). On the mock build they use
# the mock NvBufSurface, are fully TSan-clean, and stay in the default suite.
hwlib_suite = build_mock ? [] : ['nvidia_hwlib']
test('nvmm_buffer', test_buffer, protocol : 'exitcode', suite : hwlib_suite)
test('nvmm_transform', test_transform, protocol : 'exitcode', suite : hwlib_suite)
test('gst_nvmm_allocator', test_gst_allocator, protocol : 'exitcode')
# Fuzz the consumer's shm-header validation. Standalone /dev/urandom driver;
# a clean exit under ASan/UBSan is the pass. Not in the 'plugin' suite, so it
# also runs under TSan (see scripts/run-sanitizers.sh).
test('fuzz_shm_header', test_fuzz_shm, protocol : 'exitcode', args : ['200000'])
test('optical_flow_meta', test_ofa_meta, protocol : 'exitcode')
test('nvmm_det_meta', test_det_meta, protocol : 'exitcode')
test('nvmm_track_meta', test_track_meta, protocol : 'exitcode')
test('kalman_box', test_kalman_box, protocol : 'exitcode')
test('samurai_view', test_samurai_view, protocol : 'exitcode')
test('vit_grid', test_vit_grid, protocol : 'exitcode')
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

# These tests dlopen the plugin .so via the GStreamer factory, so they go in
# the 'plugin' suite — excluded under TSan (a sanitized .so can't be loaded by
# the unsanitized plugin scanner). See scripts/run-sanitizers.sh.
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

# --- Reusable motion-analytics components (analytics/) -------------------------
# All gated behind the single (default-disabled) `analytics` feature. The headers
# are always available (analytics_inc); only these tests are opt-in. Pure C++14 —
# no OpenCV anywhere in this suite.
if have_analytics
  foreach t : ['persistence_gate', 'dual_homography', 'low_texture_motion',
               'active_region', 'detection_motion_gate', 'motion_magnify']
    analytics_exe = executable('test_analytics_' + t, 'test_analytics_' + t + '.cpp',
      include_directories : [analytics_inc],
    )
    test('analytics_' + t, analytics_exe, protocol : 'exitcode')
  endforeach
endif

# --- Golden comparisons: hand-rolled analytics vs OpenCV as reference oracle ----
# The only OpenCV user in the tree, behind its own default-disabled feature.
# Fused-stage tests pin the numerics (tolerances documented per stage in the
# test files); component tests compare end-to-end behaviour on the same scenes.
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

# Parity-checks the CUDA low_texture_motion against the (golden-validated) host
# implementation on deterministic synthetic scenes — self-contained. Real-GPU
# test, tagged nvidia_hwlib like the samurai probe.
if have_analytics_cuda
  analytics_kernel_probe = executable('analytics_kernel_probe',
    ['analytics_kernel_probe.cpp', '../analytics/analytics_kernels.cu'],
    include_directories : [analytics_inc],
    dependencies : [analytics_cuda_dep],
    cuda_args : analytics_cuda_args,
    # Link via g++, not nvcc: meson's CUDA linker for a mixed C++/CUDA target
    # emits -Wl,-rpath unwrapped and nvcc rejects it. cudart arrives via
    # analytics_cuda_dep; the kernels use no relocatable device code.
    link_language : 'cpp',
  )
  test('analytics_kernels', analytics_kernel_probe, protocol : 'exitcode',
    suite : 'nvidia_hwlib')
endif
