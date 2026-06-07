# Single source for both modes: the .cpp files pick the mock vs real
# NvBufSurface header via NVMM_MOCK_API (see their #include guards).
# nvmm_det_meta.cpp is the DeepStream-free GstNvmmDetMeta carrying detections
# across the IPC boundary (its optional NvDs bridge is #ifdef NVMM_DEEPSTREAM_META).
add_library(nvmm_common STATIC nvmm_buffer.cpp nvmm_transform.cpp nvmm_det_meta.cpp)

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

if(HAVE_DEEPSTREAM_META)
    target_compile_definitions(nvmm_common PUBLIC NVMM_DEEPSTREAM_META)
    target_link_libraries(nvmm_common PUBLIC nvmm::nvds_meta)
endif()
