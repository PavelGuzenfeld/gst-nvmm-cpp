add_executable(bench_nvmm bench_nvmm.cpp)

target_link_libraries(bench_nvmm PRIVATE nvmm_common)

if(BUILD_MOCK)
    target_compile_definitions(bench_nvmm PRIVATE NVMM_MOCK_API)
endif()
