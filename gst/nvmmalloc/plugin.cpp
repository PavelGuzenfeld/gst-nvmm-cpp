#include "gstnvmmallocator.h"

#include <gst/gst.h>

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

static gboolean plugin_init(GstPlugin* plugin) {
    (void)plugin;
    /* Register the allocator type — this makes it discoverable */
    GstAllocator* alloc = gst_nvmm_allocator_new(0);
    gst_allocator_register(GST_NVMM_MEMORY_TYPE, alloc);
    return TRUE;
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvmmalloc,
    "NVMM memory allocator for Jetson NvBufSurface",
    plugin_init,
    "1.0.0",
    "LGPL",
    "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp"
)
