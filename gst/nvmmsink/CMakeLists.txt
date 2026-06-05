find_package(Threads REQUIRED)

add_library(gstnvmmsink SHARED
    gstnvmmsink.cpp
)

target_include_directories(gstnvmmsink
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/gst/nvmmalloc
)

target_link_libraries(gstnvmmsink
    PUBLIC
        nvmm_common
        gstnvmmalloc
        PkgConfig::GSTREAMER_VIDEO
        Threads::Threads
        rt
)

if(BUILD_MOCK)
    target_compile_definitions(gstnvmmsink PRIVATE NVMM_MOCK_API)
endif()

install(TARGETS gstnvmmsink
    LIBRARY DESTINATION ${GSTREAMER_PLUGINS_DIR}
)
