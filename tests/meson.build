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

# Tests below use gst_element_factory_make, which dlopens our plugin .so
# files. Under TSan that fails (GStreamer rejects dlopen'ing TSan-instrumented
# .so files from an unsanitized plugin scanner, and TSan can't be LD_PRELOADed
# the way ASan can). Group them under the 'plugin-factory' suite so the TSan
# runner can --no-suite=plugin-factory them out.
test('nvmm_sink', test_sink, protocol : 'exitcode',
  suite : 'plugin-factory',
  is_parallel : false,
  env : ['GST_PLUGIN_PATH=' + plugin_path])
test('nvmm_appsrc', test_appsrc, protocol : 'exitcode',
  suite : 'plugin-factory',
  is_parallel : false,
  env : ['GST_PLUGIN_PATH=' + plugin_path])
test('gstcheck_elements', test_gstcheck, protocol : 'exitcode',
  suite : 'plugin-factory',
  is_parallel : false,
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
  dependencies : [gstnvmmalloc_dep, gstnvmmsink_dep, gstnvmmappsrc_dep, gst_dep, gst_video_dep],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test('integration', test_integration, protocol : 'exitcode',
  suite : 'plugin-factory',
  is_parallel : false,
  env : ['GST_PLUGIN_PATH=' + plugin_path_with_convert])

# Backend concurrency / TSan test — drives ipc_backend directly so it
# works under TSan (plugin-factory tests can't, TSan cannot be LD_PRELOADed).
test_backend_concurrency = executable('test_backend_concurrency',
  'test_backend_concurrency.cpp',
  include_directories : [
    config_inc, common_inc,
    include_directories('../gst/nvmmalloc'),
  ],
  dependencies : [
    ipc_backend_dep, gstnvmmalloc_dep, gst_dep, gst_video_dep,
    dependency('threads'),
  ],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)

test('backend_concurrency', test_backend_concurrency, protocol : 'exitcode',
  is_parallel : false,
  timeout : 30)

# Header-parsing fuzz harness. Runs a fixed 10k iterations by default,
# each iteration fed from /dev/urandom. Any ASan/UBSan/TSan finding
# during the run fails the test; otherwise the header-parsing code
# survived the random-byte stream.
fuzz_shm = executable('fuzz_shm_header',
  'fuzz_shm_header.cpp',
  include_directories : [config_inc, common_inc],
  cpp_args : build_mock ? ['-DNVMM_MOCK_API'] : [],
)
test('fuzz_shm_header', fuzz_shm, protocol : 'exitcode',
  args : ['10000'],
  timeout : 60)
