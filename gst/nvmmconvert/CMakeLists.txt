add_library(gstnvmmconvert SHARED
    gstnvmmconvert.cpp
    plugin.cpp
)

target_include_directories(gstnvmmconvert
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/gst/nvmmalloc
)

target_link_libraries(gstnvmmconvert
    PRIVATE
        nvmm_common
        gstnvmmalloc
        PkgConfig::GSTREAMER_VIDEO
)

if(BUILD_MOCK)
    target_compile_definitions(gstnvmmconvert PRIVATE NVMM_MOCK_API)
endif()

install(TARGETS gstnvmmconvert
    LIBRARY DESTINATION ${GSTREAMER_PLUGINS_DIR}
)
