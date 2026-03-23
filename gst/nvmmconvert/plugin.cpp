#include "gstnvmmconvert.h"

#include <gst/gst.h>

#ifndef PACKAGE
#define PACKAGE "gst-nvmm-cpp"
#endif

static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(plugin, "nvmmconvert",
                                GST_RANK_NONE, GST_TYPE_NVMM_CONVERT);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvmmconvert,
    "NVMM video crop/scale/convert using Tegra VIC",
    plugin_init,
    "1.0.0",
    "LGPL",
    "gst-nvmm-cpp",
    "https://github.com/PavelGuzenfeld/gst-nvmm-cpp"
)
