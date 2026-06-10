#include "gstnvmmallocator.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include "nvmm_buffer.hpp"
#include "nvmm_types.hpp"

#include <memory>

namespace {

/// Internal memory subclass that wraps an NvmmBuffer.
///
/// Only the *owner* memory holds the NvmmBuffer; a shared memory (created by
/// mem_share for zero-copy tee fan-out) carries a null `buffer` and reaches the
/// surface through its parent. The owner is kept alive by GStreamer ref-counting
/// the parent for the share's lifetime.
struct GstNvmmMemory {
    GstMemory parent;
    std::unique_ptr<nvmm::NvmmBuffer> buffer;
};

/// Walk to the root memory that actually owns the NvmmBuffer.
GstNvmmMemory* nvmm_owner(GstMemory* mem) {
    while (mem->parent) mem = mem->parent;
    return reinterpret_cast<GstNvmmMemory*>(mem);
}

}  // namespace

struct _GstNvmmAllocatorPrivate {
    nvmm::MemoryType mem_type;
};

G_DEFINE_TYPE_WITH_PRIVATE(GstNvmmAllocator, gst_nvmm_allocator, GST_TYPE_ALLOCATOR)

/* No GstAllocator::alloc(size) override — video allocators use a custom
   alloc function with explicit format/dimensions instead. See GstGLMemory,
   GstVulkanImageMemory for the upstream pattern. Use
   gst_nvmm_allocator_alloc_video() or the buffer pool. */

static void gst_nvmm_allocator_free(GstAllocator* allocator, GstMemory* memory) {
    (void)allocator;
    auto* mem = reinterpret_cast<GstNvmmMemory*>(memory);
    delete mem;
}

static gpointer gst_nvmm_allocator_mem_map(GstMemory* memory, gsize maxsize,
                                             GstMapFlags flags) {
    (void)maxsize;
    (void)flags;
    /* NVIDIA convention: mapped data = NvBufSurface*.
       This is NOT a CPU-accessible pixel pointer. NVIDIA elements
       (nvvidconv, nvv4l2decoder etc.) cast this back to NvBufSurface*
       to access the hardware buffer.
       For actual CPU pixel access, use gst_nvmm_memory_map_plane(). */
    auto* mem = nvmm_owner(memory);
    if (!mem->buffer) return nullptr;
    return mem->buffer->raw();
}

static void gst_nvmm_allocator_mem_unmap(GstMemory* memory) {
    (void)memory;
}

static GstMemory* gst_nvmm_allocator_mem_share(GstMemory* memory,
                                               gssize offset, gssize size) {
    /* Share by reference. NVMM memory is an opaque NvBufSurface: surface access
       always goes through the owner (nvmm_owner) and returns the whole surface,
       so byte sub-regions don't sub-divide pixels. offset/size are still threaded
       into gst_memory_init below purely for GstMemory size accounting (keep them
       — don't "simplify" to 0/maxsize, or gst_buffer_resize math breaks).
       This keeps tee fan-out and make_writable zero-copy; safe because consumers
       read device pixels only. The owner is kept alive: gst_memory_init refs the
       parent for the share's lifetime, and the core unrefs it on free.
       The share is READONLY so a write through one branch can't silently mutate
       the surface every other branch sees.

       No mem_copy override on purpose: a real writable copy needs a device-side
       NvBufSurface copy (NvBufSurfaceCopy/NPP), out of this phase's scope. The
       core's fallback mem_copy would copy the NvBufSurface* handle (what mem_map
       returns), not pixels — so do NOT rely on gst_memory_copy for NVMM memory.
       It's unreachable today: shares are READONLY and nothing copies NVMM. */
    GstNvmmMemory* owner = nvmm_owner(memory);
    GstMemory* parent = GST_MEMORY_CAST(owner);
    if (size == -1) size = static_cast<gssize>(memory->size) - offset;

    auto* shared = new GstNvmmMemory{};  /* null buffer: references the owner */
    auto flags = static_cast<GstMemoryFlags>(
        GST_MINI_OBJECT_FLAGS(parent) | GST_MEMORY_FLAG_READONLY);
    gst_memory_init(GST_MEMORY_CAST(shared), flags, memory->allocator, parent,
                    memory->maxsize, memory->align,
                    memory->offset + offset, static_cast<gsize>(size));
    return GST_MEMORY_CAST(shared);
}

static void gst_nvmm_allocator_class_init(GstNvmmAllocatorClass* klass) {
    auto* allocator_class = GST_ALLOCATOR_CLASS(klass);
    allocator_class->alloc = nullptr;  /* use gst_nvmm_allocator_alloc_video() */
    allocator_class->free = gst_nvmm_allocator_free;
}

static void gst_nvmm_allocator_init(GstNvmmAllocator* self) {
    GstAllocator* alloc = GST_ALLOCATOR(self);
    self->priv = static_cast<GstNvmmAllocatorPrivate*>(
        gst_nvmm_allocator_get_instance_private(self));
    self->priv->mem_type = nvmm::MemoryType::kSurfaceArray;

    alloc->mem_type = GST_NVMM_MEMORY_TYPE;
    alloc->mem_map = gst_nvmm_allocator_mem_map;
    alloc->mem_unmap = gst_nvmm_allocator_mem_unmap;
    alloc->mem_share = gst_nvmm_allocator_mem_share;

    GST_OBJECT_FLAG_SET(alloc, GST_ALLOCATOR_FLAG_CUSTOM_ALLOC);
}

/* Public API */

GstAllocator* gst_nvmm_allocator_new(int mem_type) {
    auto* alloc = GST_NVMM_ALLOCATOR(g_object_new(GST_TYPE_NVMM_ALLOCATOR, nullptr));
    alloc->priv->mem_type = static_cast<nvmm::MemoryType>(mem_type);
    return GST_ALLOCATOR(alloc);
}

gboolean gst_is_nvmm_memory(GstMemory* mem) {
    return mem && mem->allocator &&
           g_strcmp0(mem->allocator->mem_type, GST_NVMM_MEMORY_TYPE) == 0;
}

void* gst_nvmm_memory_get_surface(GstMemory* mem) {
    if (!gst_is_nvmm_memory(mem)) return nullptr;
    auto* nvmm_mem = nvmm_owner(mem);
    return nvmm_mem->buffer ? nvmm_mem->buffer->raw() : nullptr;
}

gboolean gst_nvmm_memory_map_plane(GstMemory* mem, guint plane,
                                    GstMapFlags flags,
                                    guint8** data, gsize* size) {
    if (!gst_is_nvmm_memory(mem) || !data || !size) return FALSE;

    auto* nvmm_mem = nvmm_owner(mem);
    if (!nvmm_mem->buffer) return FALSE;

    nvmm::Result<nvmm::ByteSpan> result =
        (flags & GST_MAP_WRITE)
            ? nvmm_mem->buffer->map_write(plane)
            : nvmm_mem->buffer->map_read(plane);

    if (!result) {
        GST_ERROR("NVMM plane %u map failed: %s", plane,
                  result.error().detail.c_str());
        *data = nullptr;
        *size = 0;
        return FALSE;
    }

    *data = result.value().data();
    *size = result.value().size();
    return TRUE;
}

void gst_nvmm_memory_unmap_plane(GstMemory* mem) {
    if (!gst_is_nvmm_memory(mem)) return;
    auto* nvmm_mem = nvmm_owner(mem);
    if (nvmm_mem->buffer) {
        nvmm_mem->buffer->unmap();
    }
}

GstMemory* gst_nvmm_allocator_alloc_video(GstAllocator* allocator,
                                           int format,
                                           guint width, guint height) {
    if (!GST_IS_NVMM_ALLOCATOR(allocator) || width == 0 || height == 0) {
        return nullptr;
    }

    auto* self = GST_NVMM_ALLOCATOR(allocator);

    nvmm::SurfaceParams sp;
    sp.width = width;
    sp.height = height;
    sp.mem_type = self->priv->mem_type;

    switch (static_cast<GstVideoFormat>(format)) {
        case GST_VIDEO_FORMAT_RGBA: sp.color_format = nvmm::ColorFormat::kRGBA; break;
        case GST_VIDEO_FORMAT_BGRA: sp.color_format = nvmm::ColorFormat::kBGRA; break;
        case GST_VIDEO_FORMAT_I420: sp.color_format = nvmm::ColorFormat::kI420; break;
        case GST_VIDEO_FORMAT_NV21: sp.color_format = nvmm::ColorFormat::kNV21; break;
        default:                    sp.color_format = nvmm::ColorFormat::kNV12; break;
    }

    auto result = nvmm::NvmmBuffer::create(sp);
    if (!result) {
        GST_ERROR_OBJECT(allocator, "Failed to create NVMM buffer %ux%u: %s",
                         width, height, result.error().detail.c_str());
        return nullptr;
    }

    auto* mem = new GstNvmmMemory{};
    auto actual_size = static_cast<gsize>((*result).data_size());

    /* No NO_SHARE: the memory is share-capable (mem_share above), so tee
       fan-out + make_writable reference the same NvBufSurface instead of
       triggering a (broken/expensive) deep copy. */
    gst_memory_init(GST_MEMORY_CAST(mem),
                    static_cast<GstMemoryFlags>(0),
                    allocator, nullptr, actual_size, 0, 0, actual_size);

    mem->buffer = std::make_unique<nvmm::NvmmBuffer>(std::move(*result));
    return GST_MEMORY_CAST(mem);
}
