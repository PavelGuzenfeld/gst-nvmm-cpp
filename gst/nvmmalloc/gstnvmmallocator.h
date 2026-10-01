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

typedef struct _GstNvmmAllocatorPrivate GstNvmmAllocatorPrivate;

struct _GstNvmmAllocator {
    GstAllocator parent;
    GstNvmmAllocatorPrivate* priv;
};

struct _GstNvmmAllocatorClass {
    GstAllocatorClass parent_class;
};

GType gst_nvmm_allocator_get_type(void);

/// `mem_type`: 0 = default, 4 = surface array (Jetson), 6 = system (mock/test).
GstAllocator* gst_nvmm_allocator_new(int mem_type);

/// The allocator has no alloc(size) path; use this or the buffer pool.
GstMemory* gst_nvmm_allocator_alloc_video(GstAllocator* allocator,
                                           int format,
                                           guint width, guint height);

gboolean gst_is_nvmm_memory(GstMemory* mem);

void* gst_nvmm_memory_get_surface(GstMemory* mem);

/// For CPU access: gst_memory_map() returns the NvBufSurface*, and planes are
/// not contiguous on SURFACE_ARRAY.
gboolean gst_nvmm_memory_map_plane(GstMemory* mem, guint plane,
                                    GstMapFlags flags,
                                    guint8** data, gsize* size);

void gst_nvmm_memory_unmap_plane(GstMemory* mem);

G_END_DECLS
