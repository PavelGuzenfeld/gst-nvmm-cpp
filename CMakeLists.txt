cmake_minimum_required(VERSION 3.16)
project(gst-nvmm-cpp
    VERSION 1.1.1
    LANGUAGES C CXX
)

# ---- Compile settings (match meson: c++14, c11, -Wall -Wextra -Werror, RelWithDebInfo) ----
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WERROR "Treat compiler warnings as errors (matches meson's werror=true default)" ON)

add_compile_options(-Wall -Wextra)
if(WERROR)
    add_compile_options(-Werror)
endif()

include(GNUInstallDirs)

# ---- GStreamer deps via pkg-config ----
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER        REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.16.0)
pkg_check_modules(GSTREAMER_BASE   REQUIRED IMPORTED_TARGET gstreamer-base-1.0>=1.16.0)
pkg_check_modules(GSTREAMER_VIDEO  REQUIRED IMPORTED_TARGET gstreamer-video-1.0>=1.16.0)
pkg_check_modules(GSTREAMER_ALLOC  REQUIRED IMPORTED_TARGET gstreamer-allocators-1.0>=1.16.0)

# Install plugins where GStreamer actually looks for them, so `sudo cmake --install`
# is enough — no GST_PLUGIN_PATH export needed. Overridable with -DGSTREAMER_PLUGINS_DIR=...
pkg_get_variable(_gst_pluginsdir gstreamer-1.0 pluginsdir)
if(NOT _gst_pluginsdir)
    set(_gst_pluginsdir "${CMAKE_INSTALL_FULL_LIBDIR}/gstreamer-1.0")
endif()
set(GSTREAMER_PLUGINS_DIR "${_gst_pluginsdir}" CACHE PATH
    "Install destination for GStreamer plugins (defaults to system pluginsdir from pkg-config)")

# RPATH so installed plugins resolve their shared-lib siblings in the same dir
# without LD_LIBRARY_PATH.
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# ---- NVIDIA Jetson libs (NvBufSurface / NvBufSurfTransform) ----
# Try pkg-config first, then fall back to standard Jetson paths.
set(NVMM_LIB_DIRS
    /usr/lib/aarch64-linux-gnu/tegra
    /usr/lib/aarch64-linux-gnu/nvidia
    /opt/nvidia/deepstream/deepstream/lib
    /usr/lib/nvidia
    /usr/lib/aarch64-linux-gnu
)
set(NVMM_INC_CANDIDATES
    /usr/src/jetson_multimedia_api/include
    /usr/include/nvbufsurface
    /opt/nvidia/deepstream/deepstream/sources/includes
)

# Helper: try pkg-config (quietly), fall back to find_library in Jetson dirs.
# Sets <upper>_FOUND_VIA ("pkgconfig" | "fallback" | "none") and <upper>_WHERE.
function(_nvmm_find_lib upper name)
    pkg_check_modules(${upper} QUIET IMPORTED_TARGET ${name})
    if(${upper}_FOUND)
        set(${upper}_FOUND_VIA "pkgconfig" PARENT_SCOPE)
        set(${upper}_WHERE "pkg-config" PARENT_SCOPE)
        return()
    endif()
    find_library(${upper}_LIB ${name} PATHS ${NVMM_LIB_DIRS} NO_DEFAULT_PATH)
    if(NOT ${upper}_LIB)
        find_library(${upper}_LIB ${name})
    endif()
    if(${upper}_LIB)
        set(${upper}_LIB "${${upper}_LIB}" PARENT_SCOPE)
        set(${upper}_FOUND_VIA "fallback" PARENT_SCOPE)
        set(${upper}_WHERE "${${upper}_LIB}" PARENT_SCOPE)
    else()
        set(${upper}_FOUND_VIA "none" PARENT_SCOPE)
        set(${upper}_WHERE "not found" PARENT_SCOPE)
    endif()
endfunction()

_nvmm_find_lib(NVBUFSURFACE      nvbufsurface)
_nvmm_find_lib(NVBUFSURFTRANSFORM nvbufsurftransform)

# Locate the NvBufSurface headers
set(NVMM_INC_DIRS "")
foreach(d ${NVMM_INC_CANDIDATES})
    if(IS_DIRECTORY "${d}")
        list(APPEND NVMM_INC_DIRS "${d}")
    endif()
endforeach()

if(NOT NVBUFSURFACE_FOUND_VIA STREQUAL "none" AND NVMM_INC_DIRS)
    set(HAVE_NVBUFSURFACE ON)
else()
    set(HAVE_NVBUFSURFACE OFF)
endif()

if(NOT NVBUFSURFTRANSFORM_FOUND_VIA STREQUAL "none")
    set(HAVE_NVBUFSURFTRANSFORM ON)
else()
    set(HAVE_NVBUFSURFTRANSFORM OFF)
endif()

# Build imported targets so downstream just links to nvmm::*
if(HAVE_NVBUFSURFACE AND NOT TARGET nvmm::nvbufsurface)
    add_library(nvmm::nvbufsurface INTERFACE IMPORTED)
    if(NVBUFSURFACE_FOUND_VIA STREQUAL "pkgconfig")
        target_link_libraries(nvmm::nvbufsurface INTERFACE PkgConfig::NVBUFSURFACE)
    else()
        target_link_libraries(nvmm::nvbufsurface INTERFACE "${NVBUFSURFACE_LIB}")
    endif()
    target_include_directories(nvmm::nvbufsurface SYSTEM INTERFACE ${NVMM_INC_DIRS})
endif()

if(HAVE_NVBUFSURFTRANSFORM AND NOT TARGET nvmm::nvbufsurftransform)
    add_library(nvmm::nvbufsurftransform INTERFACE IMPORTED)
    if(NVBUFSURFTRANSFORM_FOUND_VIA STREQUAL "pkgconfig")
        target_link_libraries(nvmm::nvbufsurftransform INTERFACE PkgConfig::NVBUFSURFTRANSFORM)
    else()
        target_link_libraries(nvmm::nvbufsurftransform INTERFACE "${NVBUFSURFTRANSFORM_LIB}")
    endif()
endif()

# ---- JetPack detection via /etc/nv_tegra_release ----
set(JETPACK_VERSION "unknown")
if(EXISTS "/etc/nv_tegra_release")
    file(READ "/etc/nv_tegra_release" _l4t_release LIMIT 256)
    string(REGEX MATCH "R([0-9]+)" _l4t_match "${_l4t_release}")
    if(_l4t_match)
        set(_l4t_major "${CMAKE_MATCH_1}")
        if(_l4t_major STREQUAL "35")
            set(JETPACK_VERSION "5")
        elseif(_l4t_major STREQUAL "36")
            set(JETPACK_VERSION "6")
        endif()
    endif()
endif()

# ---- NvSciBuf (JP6+ only) ----
set(HAVE_NVSCIBUF OFF)
set(NVSCIBUF_WHERE "skipped (not JP6)")
if(JETPACK_VERSION STREQUAL "6")
    _nvmm_find_lib(NVSCIBUF nvscibuf)
    if(NOT NVSCIBUF_FOUND_VIA STREQUAL "none")
        set(HAVE_NVSCIBUF ON)
    endif()
endif()

# ---- Mock fallback when NVMM libs are missing ----
if(HAVE_NVBUFSURFACE)
    set(BUILD_MOCK OFF)
    set(_build_mode "real (NvBufSurface)")
else()
    set(BUILD_MOCK ON)
    set(_build_mode "mock API (NvBufSurface not found)")
endif()

# ---- Generated config.h ----
set(PACKAGE_VERSION "${PROJECT_VERSION}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config.h"
    @ONLY
)

# ---- Common include dirs exported to subdirs ----
set(NVMM_CONFIG_INC "${CMAKE_CURRENT_BINARY_DIR}")
set(NVMM_COMMON_INC "${CMAKE_CURRENT_SOURCE_DIR}/gst/common")

# ---- Subdirectories ----
add_subdirectory(gst/common)
add_subdirectory(gst/nvmmalloc)
add_subdirectory(gst/nvmmconvert)
add_subdirectory(gst/nvmmsink)
add_subdirectory(gst/nvmmappsrc)

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)

message(STATUS "gst-nvmm-cpp configuration summary:")
message(STATUS "  Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "  JetPack version:    ${JETPACK_VERSION}")
message(STATUS "  Build mode:         ${_build_mode}")
message(STATUS "  nvbufsurface:       ${NVBUFSURFACE_WHERE}")
message(STATUS "  nvbufsurftransform: ${NVBUFSURFTRANSFORM_WHERE}")
message(STATUS "  nvscibuf:           ${NVSCIBUF_WHERE}")
if(NVMM_INC_DIRS)
    message(STATUS "  NVMM headers:       ${NVMM_INC_DIRS}")
endif()
message(STATUS "  Plugins install to: ${GSTREAMER_PLUGINS_DIR}")
