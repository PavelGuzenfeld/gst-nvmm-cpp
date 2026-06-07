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

test('nvmm_buffer', test_buffer, protocol : 'exitcode')
test('nvmm_transform', test_transform, protocol : 'exitcode')
test('gst_nvmm_allocator', test_gst_allocator, protocol : 'exitcode')
# Fuzz the consumer's shm-header validation. Standalone /dev/urandom driver;
# a clean exit under ASan/UBSan is the pass. Not in the 'plugin' suite, so it
# also runs under TSan (see scripts/run-sanitizers.sh).
test('fuzz_shm_header', test_fuzz_shm, protocol : 'exitcode', args : ['200000'])
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
