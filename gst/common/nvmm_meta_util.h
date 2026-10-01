#pragma once

#include <gst/gst.h>

G_BEGIN_DECLS

/// gst_meta_api_type_register() is not idempotent and a second static copy of
/// this lib may exist, so reuse an existing type and re-look-up after losing a
/// race (register returns 0 on a duplicate name). Callers still wrap it in g_once.
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
