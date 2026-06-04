find_package(Threads REQUIRED)

# CUDA runtime (optional — only needed on real Jetson, not in mock mode)
find_library(CUDART_LIB cudart PATHS /usr/local/cuda/lib64)
find_path(CUDART_INC cuda_runtime.h PATHS /usr/local/cuda/include)

add_library(gstnvmmsink SHARED
    gstnvmmsink.cpp
)

target_include_directories(gstnvmmsink
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/gst/nvmmalloc
)
if(CUDART_INC)
    target_include_directories(gstnvmmsink PRIVATE ${CUDART_INC})
endif()

target_link_libraries(gstnvmmsink
    PUBLIC
        nvmm_common
        gstnvmmalloc
        PkgConfig::GSTREAMER_VIDEO
        Threads::Threads
        rt
)

if(CUDART_LIB)
    target_link_libraries(gstnvmmsink PRIVATE ${CUDART_LIB})
endif()

if(BUILD_MOCK)
    target_compile_definitions(gstnvmmsink PRIVATE NVMM_MOCK_API)
endif()

install(TARGETS gstnvmmsink
    LIBRARY DESTINATION ${GSTREAMER_PLUGINS_DIR}
)
