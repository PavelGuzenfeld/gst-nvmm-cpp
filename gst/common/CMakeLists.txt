# Single source for both modes: the .cpp files pick the mock vs real
# NvBufSurface header via NVMM_MOCK_API (see their #include guards).
add_library(nvmm_common STATIC nvmm_buffer.cpp nvmm_transform.cpp)

target_include_directories(nvmm_common
    PUBLIC
        ${NVMM_CONFIG_INC}
        ${NVMM_COMMON_INC}
)

target_link_libraries(nvmm_common
    PUBLIC
        PkgConfig::GSTREAMER
        PkgConfig::GSTREAMER_BASE
        PkgConfig::GSTREAMER_VIDEO
        PkgConfig::GSTREAMER_ALLOC
)

if(BUILD_MOCK)
    target_compile_definitions(nvmm_common PUBLIC NVMM_MOCK_API)
else()
    target_link_libraries(nvmm_common
        PUBLIC
            nvmm::nvbufsurface
            nvmm::nvbufsurftransform
    )
endif()
