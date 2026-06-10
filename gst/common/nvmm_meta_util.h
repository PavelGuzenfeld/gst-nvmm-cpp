/// Shared GstMeta API-type registration for the nvmm metas.
#pragma once

#include <gst/gst.h>

G_BEGIN_DECLS

/// Register (or reuse) a meta API GType by name. Defensive: nvmm_common is a
/// shared lib in-tree (one copy/process), but a test harness or out-of-tree
/// consumer may still static-link a second copy. gst_meta_api_type_register()
/// is NOT idempotent, so reuse an existing registration; if we still lose a
/// concurrent race, re-look-up the winner's type (the register call returns 0
/// on a duplicate name). Callers keep their own g_once around this.
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
