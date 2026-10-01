#pragma once

#include <gst/gst.h>

G_BEGIN_DECLS

static inline GType
nvmm_meta_api_register_once(const gchar *name, const gchar **tags)
{
    GType t = g_type_from_name(name);
    if (t == 0)
        t = gst_meta_api_type_register(name, tags);
    if (t == 0)
        t = g_type_from_name(name);
    return t;
}

G_END_DECLS
