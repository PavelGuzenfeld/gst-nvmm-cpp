/// GstNvmmAllocator — GStreamer allocator for NVMM (NvBufSurface) memory.
/// C header for GObject type registration. Implementation is C++17.
#pragma once

#include <gst/gst.h>
#include <gst/allocators/allocators.h>

G_BEGIN_DECLS

#define GST_TYPE_NVMM_ALLOCATOR (gst_nvmm_allocator_get_type())
#define GST_NVMM_ALLOCATOR(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVMM_ALLOCATOR, GstNvmmAllocator))
#define GST_IS_NVMM_ALLOCATOR(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVMM_ALLOCATOR))

#define GST_NVMM_MEMORY_TYPE "nvmm"

typedef struct _GstNvmmAllocator GstNvmmAllocator;
typedef struct _GstNvmmAllocatorClass GstNvmmAllocatorClass;

/// Opaque C++ impl pointer
typedef struct _GstNvmmAllocatorPrivate GstNvmmAllocatorPrivate;

struct _GstNvmmAllocator {
    GstAllocator parent;
    GstNvmmAllocatorPrivate* priv;
};

struct _GstNvmmAllocatorClass {
    GstAllocatorClass parent_class;
};

GType gst_nvmm_allocator_get_type(void);

/// Create a new NVMM allocator with the specified memory type.
/// mem_type: 0=default, 4=surface_array (Jetson), 6=system (mock/test)
GstAllocator* gst_nvmm_allocator_new(int mem_type);

/// Check if a GstMemory was allocated by the NVMM allocator.
gboolean gst_is_nvmm_memory(GstMemory* mem);

/// Get the NvBufSurface* from a GstMemory allocated by GstNvmmAllocator.
/// Returns NULL if the memory is not NVMM.
void* gst_nvmm_memory_get_surface(GstMemory* mem);

/// Map a specific plane of NVMM memory for CPU access.
/// NVMM memory is NOT directly mappable via gst_memory_map() because
/// planes are not contiguous on Jetson SURFACE_ARRAY. Use this instead.
/// @param mem    GstMemory allocated by GstNvmmAllocator
/// @param plane  Plane index (0 for Y, 1 for UV in NV12, etc.)
/// @param flags  GST_MAP_READ or GST_MAP_WRITE
/// @param data   [out] Pointer to mapped plane data
/// @param size   [out] Size of the mapped plane in bytes
/// @return TRUE on success, FALSE on error
gboolean gst_nvmm_memory_map_plane(GstMemory* mem, guint plane,
                                    GstMapFlags flags,
                                    guint8** data, gsize* size);

/// Unmap previously mapped NVMM memory planes.
void gst_nvmm_memory_unmap_plane(GstMemory* mem);

G_END_DECLS
