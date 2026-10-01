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

nvmm_add_test(test_nvmm_surface
    SOURCES test_nvmm_surface.cpp
    LIBRARIES nvmm_common
)

nvmm_add_test(test_gst_meta_and_allocator
    SOURCES test_gst_meta_and_allocator.cpp
        ${CMAKE_SOURCE_DIR}/gst/common/nvmm_optical_flow_meta.cpp
        ${CMAKE_SOURCE_DIR}/gst/common/nvmm_track_meta.cpp
        ${CMAKE_SOURCE_DIR}/gst/common/nvmm_class_meta.cpp
    LIBRARIES nvmm_common gstnvmmalloc PkgConfig::GSTREAMER
)

nvmm_add_test(test_nvmm_sink_appsrc
    SOURCES test_nvmm_sink_appsrc.cpp
    LIBRARIES gstnvmmsink gstnvmmappsrc PkgConfig::GSTREAMER
    ENV "GST_PLUGIN_PATH=${_plugin_path}"
)

nvmm_add_test(test_integration
    SOURCES test_integration.cpp
    LIBRARIES gstnvmmsink gstnvmmappsrc PkgConfig::GSTREAMER PkgConfig::GSTREAMER_VIDEO
    ENV "GST_PLUGIN_PATH=${_plugin_path_with_convert}"
)

# They share /dev/shm.
set_tests_properties(test_nvmm_sink_appsrc test_integration
    PROPERTIES RUN_SERIAL TRUE
)
