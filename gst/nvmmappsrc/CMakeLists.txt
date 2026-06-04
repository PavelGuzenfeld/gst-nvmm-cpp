add_library(gstnvmmappsrc SHARED
    gstnvmmappsrc.cpp
)

target_include_directories(gstnvmmappsrc
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/gst/nvmmalloc
)

target_link_libraries(gstnvmmappsrc
    PUBLIC
        nvmm_common
        gstnvmmalloc
        PkgConfig::GSTREAMER_VIDEO
        rt
)

if(BUILD_MOCK)
    target_compile_definitions(gstnvmmappsrc PRIVATE NVMM_MOCK_API)
endif()

install(TARGETS gstnvmmappsrc
    LIBRARY DESTINATION ${GSTREAMER_PLUGINS_DIR}
)
