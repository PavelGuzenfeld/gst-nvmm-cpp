add_library(gstnvmmalloc SHARED
    gstnvmmallocator.cpp
    gstnvmmbufferpool.cpp
    plugin.cpp
)

target_include_directories(gstnvmmalloc
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(gstnvmmalloc
    PUBLIC
        nvmm_common
        PkgConfig::GSTREAMER_ALLOC
        PkgConfig::GSTREAMER_VIDEO
)

if(BUILD_MOCK)
    target_compile_definitions(gstnvmmalloc PRIVATE NVMM_MOCK_API)
endif()

install(TARGETS gstnvmmalloc
    LIBRARY DESTINATION ${GSTREAMER_PLUGINS_DIR}
)
