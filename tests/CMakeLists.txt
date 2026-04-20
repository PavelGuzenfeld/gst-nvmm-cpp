set(_plugin_path
    "${CMAKE_BINARY_DIR}/gst/nvmmsink:${CMAKE_BINARY_DIR}/gst/nvmmappsrc:${CMAKE_BINARY_DIR}/gst/nvmmalloc"
)
set(_plugin_path_with_convert
    "${_plugin_path}:${CMAKE_BINARY_DIR}/gst/nvmmconvert"
)

function(nvmm_add_test name)
    cmake_parse_arguments(T "" "" "SOURCES;INCLUDES;LIBRARIES;ENV" ${ARGN})
    add_executable(${name} ${T_SOURCES})
    target_include_directories(${name} PRIVATE ${T_INCLUDES})
    target_link_libraries(${name} PRIVATE ${T_LIBRARIES})
    if(BUILD_MOCK)
        target_compile_definitions(${name} PRIVATE NVMM_MOCK_API)
    endif()
    add_test(NAME ${name} COMMAND ${name})
    if(T_ENV)
        set_tests_properties(${name} PROPERTIES ENVIRONMENT "${T_ENV}")
    endif()
endfunction()

nvmm_add_test(test_nvmm_buffer
    SOURCES test_nvmm_buffer.cpp
    LIBRARIES nvmm_common
)

nvmm_add_test(test_nvmm_transform
    SOURCES test_nvmm_transform.cpp
    LIBRARIES nvmm_common
)

nvmm_add_test(test_gst_nvmm_allocator
    SOURCES test_gst_nvmm_allocator.cpp
    LIBRARIES nvmm_common gstnvmmalloc PkgConfig::GSTREAMER
)

nvmm_add_test(test_nvmm_sink
    SOURCES test_nvmm_sink.cpp
    LIBRARIES gstnvmmsink PkgConfig::GSTREAMER
    ENV "GST_PLUGIN_PATH=${_plugin_path}"
)

nvmm_add_test(test_nvmm_appsrc
    SOURCES test_nvmm_appsrc.cpp
    LIBRARIES gstnvmmsink gstnvmmappsrc PkgConfig::GSTREAMER
    ENV "GST_PLUGIN_PATH=${_plugin_path}"
)

nvmm_add_test(test_gstcheck_elements
    SOURCES test_gstcheck_elements.cpp
    LIBRARIES PkgConfig::GSTREAMER PkgConfig::GSTREAMER_VIDEO
    ENV "GST_PLUGIN_PATH=${_plugin_path_with_convert}"
)

nvmm_add_test(test_integration
    SOURCES test_integration.cpp
    LIBRARIES gstnvmmsink gstnvmmappsrc PkgConfig::GSTREAMER PkgConfig::GSTREAMER_VIDEO
    ENV "GST_PLUGIN_PATH=${_plugin_path_with_convert}"
)

# Make sink/appsrc/integration tests run sequentially (they touch shared /dev/shm)
set_tests_properties(test_nvmm_sink test_nvmm_appsrc test_gstcheck_elements test_integration
    PROPERTIES RUN_SERIAL TRUE
)
