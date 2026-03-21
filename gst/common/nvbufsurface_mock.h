/// Mock NvBufSurface API for host-side compilation and testing.
/// On Jetson, the real headers from /usr/src/jetson_multimedia_api/include
/// are used instead. This mock provides the same struct layout and function
/// signatures so the plugin compiles and tests run on x86_64.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NVBUF_MAX_PLANES 4

typedef enum {
    NVBUF_MEM_DEFAULT = 0,
    NVBUF_MEM_CUDA_DEVICE,
    NVBUF_MEM_CUDA_PINNED,
    NVBUF_MEM_CUDA_UNIFIED,
    NVBUF_MEM_SURFACE_ARRAY,
    NVBUF_MEM_HANDLE,
    NVBUF_MEM_SYSTEM,
} NvBufSurfaceMemType;

typedef enum {
    NVBUF_COLOR_FORMAT_NV12 = 0,
    NVBUF_COLOR_FORMAT_RGBA,
    NVBUF_COLOR_FORMAT_BGRA,
    NVBUF_COLOR_FORMAT_I420,
    NVBUF_COLOR_FORMAT_NV21,
    NVBUF_COLOR_FORMAT_GRAY8,
} NvBufSurfaceColorFormat;

typedef enum {
    NvBufSurfTransform_None = 0,
    NvBufSurfTransform_Rotate90,
    NvBufSurfTransform_Rotate180,
    NvBufSurfTransform_Rotate270,
    NvBufSurfTransform_FlipX,
    NvBufSurfTransform_Transpose,
    NvBufSurfTransform_FlipY,
    NvBufSurfTransform_InvTranspose,
} NvBufSurfTransform_Flip;

typedef struct {
    uint32_t top;
    uint32_t left;
    uint32_t width;
    uint32_t height;
} NvBufSurfTransformRect;

typedef struct {
    NvBufSurfTransform_Flip flip;
    NvBufSurfTransformRect* src_rect;
    NvBufSurfTransformRect* dst_rect;
    uint32_t transform_flag;
} NvBufSurfTransformParams;

typedef struct {
    uint32_t num_planes;
    struct {
        uint32_t width;
        uint32_t height;
        uint32_t pitch;
        uint32_t offset;
        uint32_t psize;
        uint32_t bytesPerPix;
    } planeParams[NVBUF_MAX_PLANES]; /* renamed from the real API for simplicity */
} NvBufSurfacePlaneParamsEx;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t pitch;
    NvBufSurfaceColorFormat colorFormat;
    uint32_t layout;
    uint64_t bufferDesc;
    uint32_t dataSize;
    void* dataPtr;
    void* mappedAddr;
    int dmabuf_fd;
    NvBufSurfacePlaneParamsEx planeParams;
} NvBufSurfaceParams;

typedef struct NvBufSurface {
    uint32_t batchSize;
    uint32_t numFilled;
    uint32_t isContiguous;
    NvBufSurfaceMemType memType;
    NvBufSurfaceParams* surfaceList;
    /* mock-only bookkeeping */
    uint32_t _mock_allocated;
} NvBufSurface;

typedef struct {
    uint32_t width;
    uint32_t height;
    NvBufSurfaceColorFormat colorFormat;
    NvBufSurfaceMemType memType;
    uint32_t size;
    uint32_t layout;
    uint32_t isContiguous;
} NvBufSurfaceCreateParams;

/* ---- Mock implementations (inline for header-only usage) ---- */

static inline uint32_t _nvmm_mock_plane_count(NvBufSurfaceColorFormat fmt) {
    switch (fmt) {
        case NVBUF_COLOR_FORMAT_NV12:
        case NVBUF_COLOR_FORMAT_NV21:
            return 2;
        case NVBUF_COLOR_FORMAT_I420:
            return 3;
        default:
            return 1;
    }
}

static inline uint32_t _nvmm_mock_bpp(NvBufSurfaceColorFormat fmt) {
    switch (fmt) {
        case NVBUF_COLOR_FORMAT_RGBA:
        case NVBUF_COLOR_FORMAT_BGRA:
            return 4;
        case NVBUF_COLOR_FORMAT_GRAY8:
            return 1;
        default:
            return 1; /* per-plane bpp for planar formats */
    }
}

static inline int NvBufSurfaceCreate(NvBufSurface** surf,
                                      uint32_t batch_size,
                                      NvBufSurfaceCreateParams* params) {
    NvBufSurface* s = (NvBufSurface*)calloc(1, sizeof(NvBufSurface));
    if (!s) return -1;

    s->batchSize = batch_size;
    s->numFilled = batch_size;
    s->memType = params->memType;
    s->_mock_allocated = 1;

    s->surfaceList = (NvBufSurfaceParams*)calloc(batch_size, sizeof(NvBufSurfaceParams));
    if (!s->surfaceList) { free(s); return -1; }

    for (uint32_t i = 0; i < batch_size; i++) {
        NvBufSurfaceParams* p = &s->surfaceList[i];
        p->width = params->width;
        p->height = params->height;
        p->colorFormat = params->colorFormat;
        p->pitch = params->width * _nvmm_mock_bpp(params->colorFormat);
        p->dmabuf_fd = -1;

        uint32_t nplanes = _nvmm_mock_plane_count(params->colorFormat);
        p->planeParams.num_planes = nplanes;
        uint32_t total_size = 0;
        for (uint32_t pl = 0; pl < nplanes; pl++) {
            uint32_t pw = params->width;
            uint32_t ph = params->height;
            uint32_t bpp = _nvmm_mock_bpp(params->colorFormat);
            if (pl > 0 && (params->colorFormat == NVBUF_COLOR_FORMAT_NV12 ||
                           params->colorFormat == NVBUF_COLOR_FORMAT_NV21)) {
                ph /= 2;
            } else if (pl > 0 && params->colorFormat == NVBUF_COLOR_FORMAT_I420) {
                pw /= 2;
                ph /= 2;
            }
            p->planeParams.planeParams[pl].width = pw;
            p->planeParams.planeParams[pl].height = ph;
            p->planeParams.planeParams[pl].pitch = pw * bpp;
            p->planeParams.planeParams[pl].offset = total_size;
            p->planeParams.planeParams[pl].psize = pw * ph * bpp;
            p->planeParams.planeParams[pl].bytesPerPix = bpp;
            total_size += pw * ph * bpp;
        }

        p->dataSize = total_size;
        p->dataPtr = calloc(1, total_size);
        if (!p->dataPtr) { /* simplistic cleanup */
            free(s->surfaceList);
            free(s);
            return -1;
        }
    }

    *surf = s;
    return 0;
}

static inline int NvBufSurfaceDestroy(NvBufSurface* surf) {
    if (!surf) return -1;
    if (surf->surfaceList) {
        for (uint32_t i = 0; i < surf->batchSize; i++) {
            free(surf->surfaceList[i].dataPtr);
            free(surf->surfaceList[i].mappedAddr);
        }
        free(surf->surfaceList);
    }
    free(surf);
    return 0;
}

static inline int NvBufSurfaceMap(NvBufSurface* surf, int index,
                                   int plane, int type) {
    (void)type;
    if (!surf || index < 0 || (uint32_t)index >= surf->batchSize) return -1;
    NvBufSurfaceParams* p = &surf->surfaceList[index];
    /* In mock, mappedAddr == dataPtr */
    if (!p->mappedAddr) {
        p->mappedAddr = malloc(p->dataSize);
        if (!p->mappedAddr) return -1;
        memcpy(p->mappedAddr, p->dataPtr, p->dataSize);
    }
    (void)plane;
    return 0;
}

static inline int NvBufSurfaceUnMap(NvBufSurface* surf, int index, int plane) {
    if (!surf || index < 0 || (uint32_t)index >= surf->batchSize) return -1;
    NvBufSurfaceParams* p = &surf->surfaceList[index];
    if (p->mappedAddr) {
        memcpy(p->dataPtr, p->mappedAddr, p->dataSize);
        free(p->mappedAddr);
        p->mappedAddr = NULL;
    }
    (void)plane;
    return 0;
}

static inline int NvBufSurfaceGetFd(NvBufSurface* surf, int index, int* fd) {
    if (!surf || index < 0 || (uint32_t)index >= surf->batchSize || !fd) return -1;
    /* Mock: return a fake fd */
    *fd = 42 + index;
    surf->surfaceList[index].dmabuf_fd = *fd;
    return 0;
}

static inline int NvBufSurfaceFromFd(int fd, void** surf_ptr) {
    /* Mock: cannot really import from fd, return error */
    (void)fd;
    (void)surf_ptr;
    return -1;
}

static inline int NvBufSurfTransform(NvBufSurface* src, NvBufSurface* dst,
                                      NvBufSurfTransformParams* params) {
    if (!src || !dst || !params) return -1;
    /* Mock: just memcpy the min of src/dst data sizes */
    for (uint32_t i = 0; i < src->numFilled && i < dst->batchSize; i++) {
        uint32_t copy_size = src->surfaceList[i].dataSize;
        if (dst->surfaceList[i].dataSize < copy_size)
            copy_size = dst->surfaceList[i].dataSize;
        memcpy(dst->surfaceList[i].dataPtr, src->surfaceList[i].dataPtr, copy_size);
    }
    return 0;
}

#ifdef __cplusplus
}
#endif
