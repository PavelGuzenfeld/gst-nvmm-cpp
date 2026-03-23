#include "gstnvmmallocator.h"

#include <gst/gst.h>

#include "nvmm_buffer.hpp"
#include "nvmm_types.hpp"

#include <memory>

namespace {

/// Internal memory subclass that wraps an NvmmBuffer.
struct GstNvmmMemory {
    GstMemory parent;
    std::unique_ptr<nvmm::NvmmBuffer> buffer;
};

}  // namespace

struct _GstNvmmAllocatorPrivate {
    nvmm::MemoryType mem_type;
};

G_DEFINE_TYPE_WITH_PRIVATE(GstNvmmAllocator, gst_nvmm_allocator, GST_TYPE_ALLOCATOR)

static GstMemory* gst_nvmm_allocator_alloc(GstAllocator* allocator,
                                             gsize size,
                                             GstAllocationParams* params) {
    (void)params;
    auto* self = GST_NVMM_ALLOCATOR(allocator);

    /* Decode width/height from size — for proper usage, callers should use
       the buffer pool path. This fallback treats size as width*height NV12. */
    uint32_t height = 1080;
    uint32_t width = 1920;
    if (size > 0) {
        /* Heuristic: assume NV12 (1.5 bytes/pixel) */
        uint64_t pixels = (size * 2) / 3;
        /* Find reasonable dimensions */
        width = 1;
        height = 1;
        for (uint32_t w = 320; w <= 3840; w += 2) {
            uint32_t h = static_cast<uint32_t>((pixels + w - 1) / w);
            if (static_cast<uint64_t>(w) * h * 3 / 2 >= size) {
                width = w;
                height = h;
                break;
            }
        }
    }

    nvmm::SurfaceParams surface_params;
    surface_params.width = width;
    surface_params.height = height;
    surface_params.color_format = nvmm::ColorFormat::kNV12;
    surface_params.mem_type = self->priv->mem_type;
    surface_params.num_surfaces = 1;

    auto result = nvmm::NvmmBuffer::create(surface_params);
    if (!result) {
        GST_ERROR_OBJECT(allocator, "Failed to create NVMM buffer: %s",
                         result.error().detail.c_str());
        return nullptr;
    }

    auto* mem = new GstNvmmMemory{};
    auto actual_size = static_cast<gsize>((*result).data_size());

    /* NVMM convention: gst_memory_map returns the NvBufSurface* pointer.
       This is what NVIDIA's nvvidconv/nvv4l2 elements expect.
       For actual CPU pixel access, use gst_nvmm_memory_map_plane(). */
    gst_memory_init(GST_MEMORY_CAST(mem),
                    GST_MEMORY_FLAG_NO_SHARE,
                    allocator, nullptr, actual_size, 0, 0, actual_size);

    mem->buffer = std::make_unique<nvmm::NvmmBuffer>(std::move(*result));

    return GST_MEMORY_CAST(mem);
}

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
    auto* mem = reinterpret_cast<GstNvmmMemory*>(memory);
    if (!mem->buffer) return nullptr;
    return mem->buffer->raw();
}

static void gst_nvmm_allocator_mem_unmap(GstMemory* memory) {
    (void)memory;
}

static void gst_nvmm_allocator_class_init(GstNvmmAllocatorClass* klass) {
    auto* allocator_class = GST_ALLOCATOR_CLASS(klass);
    allocator_class->alloc = gst_nvmm_allocator_alloc;
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
    auto* nvmm_mem = reinterpret_cast<GstNvmmMemory*>(mem);
    return nvmm_mem->buffer ? nvmm_mem->buffer->raw() : nullptr;
}

gboolean gst_nvmm_memory_map_plane(GstMemory* mem, guint plane,
                                    GstMapFlags flags,
                                    guint8** data, gsize* size) {
    if (!gst_is_nvmm_memory(mem) || !data || !size) return FALSE;

    auto* nvmm_mem = reinterpret_cast<GstNvmmMemory*>(mem);
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
    auto* nvmm_mem = reinterpret_cast<GstNvmmMemory*>(mem);
    if (nvmm_mem->buffer) {
        nvmm_mem->buffer->unmap();
    }
}
