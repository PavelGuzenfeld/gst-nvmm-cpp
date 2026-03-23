/// Mock NvBufSurface API for host-side compilation and testing.
/// Struct layout matches the real NVIDIA headers exactly so that a single
/// implementation file works for both mock and real builds.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NVBUF_MAX_PLANES 4
#define STRUCTURE_PADDING 4

typedef enum {
    NVBUF_MAP_READ,
    NVBUF_MAP_WRITE,
    NVBUF_MAP_READ_WRITE,
} NvBufSurfaceMemMapFlags;

typedef enum {
    NVBUF_COLOR_FORMAT_INVALID,
    NVBUF_COLOR_FORMAT_GRAY8,
    NVBUF_COLOR_FORMAT_YUV420,    /* I420 */
    NVBUF_COLOR_FORMAT_YVU420,
    NVBUF_COLOR_FORMAT_YUV420_ER,
    NVBUF_COLOR_FORMAT_YVU420_ER,
    NVBUF_COLOR_FORMAT_NV12,      /* = 6 */
    NVBUF_COLOR_FORMAT_NV12_ER,
    NVBUF_COLOR_FORMAT_NV21,      /* = 8 */
    NVBUF_COLOR_FORMAT_NV21_ER,
    NVBUF_COLOR_FORMAT_UYVY,
    NVBUF_COLOR_FORMAT_UYVY_ER,
    NVBUF_COLOR_FORMAT_VYUY,
    NVBUF_COLOR_FORMAT_VYUY_ER,
    NVBUF_COLOR_FORMAT_YUYV,
    NVBUF_COLOR_FORMAT_YUYV_ER,
    NVBUF_COLOR_FORMAT_YVYU,
    NVBUF_COLOR_FORMAT_YVYU_ER,
    NVBUF_COLOR_FORMAT_YUV444,
    NVBUF_COLOR_FORMAT_RGBA,      /* = 19 */
    NVBUF_COLOR_FORMAT_BGRA,      /* = 20 */
    NVBUF_COLOR_FORMAT_ARGB,
    NVBUF_COLOR_FORMAT_ABGR,
    NVBUF_COLOR_FORMAT_RGBx,
    NVBUF_COLOR_FORMAT_BGRx,
    NVBUF_COLOR_FORMAT_xRGB,
    NVBUF_COLOR_FORMAT_xBGR,
    NVBUF_COLOR_FORMAT_RGB,
    NVBUF_COLOR_FORMAT_BGR,
    NVBUF_COLOR_FORMAT_LAST
} NvBufSurfaceColorFormat;

typedef enum {
    NVBUF_LAYOUT_PITCH,
    NVBUF_LAYOUT_BLOCK_LINEAR,
} NvBufSurfaceLayout;

typedef enum {
    NVBUF_MEM_DEFAULT,
    NVBUF_MEM_CUDA_PINNED,
    NVBUF_MEM_CUDA_DEVICE,
    NVBUF_MEM_CUDA_UNIFIED,
    NVBUF_MEM_SURFACE_ARRAY,
    NVBUF_MEM_HANDLE,
    NVBUF_MEM_SYSTEM,
} NvBufSurfaceMemType;

typedef enum {
    NvBufSurfTransform_None = 0,
    NvBufSurfTransform_Rotate90,
    NvBufSurfTransform_Rotate180,
    NvBufSurfTransform_Rotate270,
    NvBufSurfTransform_FlipX,
    NvBufSurfTransform_FlipY,
    NvBufSurfTransform_Transpose,
    NvBufSurfTransform_InvTranspose,
} NvBufSurfTransform_Flip;

typedef enum {
    NvBufSurfTransformError_ROI_Error = -4,
    NvBufSurfTransformError_Invalid_Params = -3,
    NvBufSurfTransformError_Execution_Error = -2,
    NvBufSurfTransformError_Unsupported = -1,
    NvBufSurfTransformError_Success = 0
} NvBufSurfTransform_Error;

typedef enum {
    NVBUFSURF_TRANSFORM_CROP_SRC  = 1,
    NVBUFSURF_TRANSFORM_CROP_DST  = 1 << 1,
    NVBUFSURF_TRANSFORM_FILTER    = 1 << 2,
    NVBUFSURF_TRANSFORM_FLIP      = 1 << 3,
} NvBufSurfTransform_Transform_Flag;

typedef enum {
    NvBufSurfTransformInter_Nearest = 0,
    NvBufSurfTransformInter_Bilinear,
    NvBufSurfTransformInter_Default = 6
} NvBufSurfTransform_Inter;

typedef struct {
    uint32_t top;
    uint32_t left;
    uint32_t width;
    uint32_t height;
} NvBufSurfTransformRect;

typedef struct {
    uint32_t transform_flag;
    NvBufSurfTransform_Flip transform_flip;
    NvBufSurfTransform_Inter transform_filter;
    NvBufSurfTransformRect* src_rect;
    NvBufSurfTransformRect* dst_rect;
} NvBufSurfTransformParams;

/* Matches real NvBufSurfacePlaneParams: flat arrays indexed by plane */
typedef struct NvBufSurfacePlaneParams {
    uint32_t num_planes;
    uint32_t width[NVBUF_MAX_PLANES];
    uint32_t height[NVBUF_MAX_PLANES];
    uint32_t pitch[NVBUF_MAX_PLANES];
    uint32_t offset[NVBUF_MAX_PLANES];
    uint32_t psize[NVBUF_MAX_PLANES];
    uint32_t bytesPerPix[NVBUF_MAX_PLANES];
    void* _reserved[STRUCTURE_PADDING * NVBUF_MAX_PLANES];
} NvBufSurfacePlaneParams;

/* Matches real NvBufSurfaceMappedAddr: per-plane void* array */
typedef struct NvBufSurfaceMappedAddr {
    void* addr[NVBUF_MAX_PLANES];
    void* eglImage;
    void* _reserved[STRUCTURE_PADDING];
} NvBufSurfaceMappedAddr;

typedef struct NvBufSurfaceCreateParams {
    uint32_t gpuId;
    uint32_t width;
    uint32_t height;
    uint32_t size;
    uint32_t isContiguous;
    NvBufSurfaceColorFormat colorFormat;
    NvBufSurfaceLayout layout;
    NvBufSurfaceMemType memType;
} NvBufSurfaceCreateParams;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t pitch;
    NvBufSurfaceColorFormat colorFormat;
    NvBufSurfaceLayout layout;
    uint64_t bufferDesc;
    uint32_t dataSize;
    void* dataPtr;
    NvBufSurfacePlaneParams planeParams;
    NvBufSurfaceMappedAddr mappedAddr;
    void* paramex;
    void* _reserved[STRUCTURE_PADDING - 1];
} NvBufSurfaceParams;

typedef struct NvBufSurface {
    uint32_t gpuId;
    uint32_t batchSize;
    uint32_t numFilled;
    uint32_t isContiguous;
    NvBufSurfaceMemType memType;
    NvBufSurfaceParams* surfaceList;
    void* _reserved[STRUCTURE_PADDING];
} NvBufSurface;

/* ---- Helpers ---- */

static inline uint32_t _mock_plane_count(NvBufSurfaceColorFormat fmt) {
    switch (fmt) {
        case NVBUF_COLOR_FORMAT_NV12:
        case NVBUF_COLOR_FORMAT_NV21:
            return 2;
        case NVBUF_COLOR_FORMAT_YUV420:
            return 3;
        default:
            return 1;
    }
}

static inline uint32_t _mock_bpp(NvBufSurfaceColorFormat fmt) {
    switch (fmt) {
        case NVBUF_COLOR_FORMAT_RGBA:
        case NVBUF_COLOR_FORMAT_BGRA:
            return 4;
        default:
            return 1;
    }
}

/* ---- Mock function implementations ---- */

static inline int NvBufSurfaceCreate(NvBufSurface** surf,
                                      uint32_t batch_size,
                                      NvBufSurfaceCreateParams* params) {
    NvBufSurface* s = (NvBufSurface*)calloc(1, sizeof(NvBufSurface));
    if (!s) return -1;

    s->batchSize = batch_size;
    s->numFilled = batch_size;
    s->memType = params->memType;

    s->surfaceList = (NvBufSurfaceParams*)calloc(batch_size, sizeof(NvBufSurfaceParams));
    if (!s->surfaceList) { free(s); return -1; }

    for (uint32_t i = 0; i < batch_size; i++) {
        NvBufSurfaceParams* p = &s->surfaceList[i];
        p->width = params->width;
        p->height = params->height;
        p->colorFormat = params->colorFormat;
        p->layout = params->layout;
        p->pitch = params->width * _mock_bpp(params->colorFormat);
        p->bufferDesc = 42 + i;  /* fake DMA-buf fd */

        uint32_t nplanes = _mock_plane_count(params->colorFormat);
        p->planeParams.num_planes = nplanes;
        uint32_t total_size = 0;
        for (uint32_t pl = 0; pl < nplanes; pl++) {
            uint32_t pw = params->width;
            uint32_t ph = params->height;
            uint32_t bpp = _mock_bpp(params->colorFormat);
            if (pl > 0 && (params->colorFormat == NVBUF_COLOR_FORMAT_NV12 ||
                           params->colorFormat == NVBUF_COLOR_FORMAT_NV21)) {
                ph /= 2;
            } else if (pl > 0 && params->colorFormat == NVBUF_COLOR_FORMAT_YUV420) {
                pw /= 2;
                ph /= 2;
            }
            p->planeParams.width[pl] = pw;
            p->planeParams.height[pl] = ph;
            p->planeParams.pitch[pl] = pw * bpp;
            p->planeParams.offset[pl] = total_size;
            p->planeParams.psize[pl] = pw * ph * bpp;
            p->planeParams.bytesPerPix[pl] = bpp;
            total_size += pw * ph * bpp;
        }

        p->dataSize = total_size;
        p->dataPtr = calloc(1, total_size);
        if (!p->dataPtr) {
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
            for (int p = 0; p < NVBUF_MAX_PLANES; p++)
                free(surf->surfaceList[i].mappedAddr.addr[p]);
        }
        free(surf->surfaceList);
    }
    free(surf);
    return 0;
}

static inline int NvBufSurfaceMap(NvBufSurface* surf, int index,
                                   int plane, NvBufSurfaceMemMapFlags type) {
    (void)type;
    if (!surf || index < 0 || (uint32_t)index >= surf->numFilled) return -1;
    NvBufSurfaceParams* p = &surf->surfaceList[index];

    if (plane < 0) {
        /* Map all planes */
        for (uint32_t pl = 0; pl < p->planeParams.num_planes; pl++) {
            if (!p->mappedAddr.addr[pl]) {
                uint32_t psize = p->planeParams.psize[pl];
                p->mappedAddr.addr[pl] = malloc(psize);
                if (!p->mappedAddr.addr[pl]) return -1;
                uint8_t* src = (uint8_t*)p->dataPtr + p->planeParams.offset[pl];
                memcpy(p->mappedAddr.addr[pl], src, psize);
            }
        }
    } else if ((uint32_t)plane < p->planeParams.num_planes) {
        if (!p->mappedAddr.addr[plane]) {
            uint32_t psize = p->planeParams.psize[plane];
            p->mappedAddr.addr[plane] = malloc(psize);
            if (!p->mappedAddr.addr[plane]) return -1;
            uint8_t* src = (uint8_t*)p->dataPtr + p->planeParams.offset[plane];
            memcpy(p->mappedAddr.addr[plane], src, psize);
        }
    }
    return 0;
}

static inline int NvBufSurfaceUnMap(NvBufSurface* surf, int index, int plane) {
    if (!surf || index < 0 || (uint32_t)index >= surf->numFilled) return -1;
    NvBufSurfaceParams* p = &surf->surfaceList[index];

    if (plane < 0) {
        for (uint32_t pl = 0; pl < p->planeParams.num_planes; pl++) {
            if (p->mappedAddr.addr[pl]) {
                uint8_t* dst = (uint8_t*)p->dataPtr + p->planeParams.offset[pl];
                memcpy(dst, p->mappedAddr.addr[pl], p->planeParams.psize[pl]);
                free(p->mappedAddr.addr[pl]);
                p->mappedAddr.addr[pl] = NULL;
            }
        }
    } else if ((uint32_t)plane < p->planeParams.num_planes) {
        if (p->mappedAddr.addr[plane]) {
            uint8_t* dst = (uint8_t*)p->dataPtr + p->planeParams.offset[plane];
            memcpy(dst, p->mappedAddr.addr[plane], p->planeParams.psize[plane]);
            free(p->mappedAddr.addr[plane]);
            p->mappedAddr.addr[plane] = NULL;
        }
    }
    return 0;
}

static inline int NvBufSurfaceSyncForCpu(NvBufSurface* surf, int index, int plane) {
    (void)surf; (void)index; (void)plane;
    return 0;  /* no-op in mock */
}

static inline int NvBufSurfaceSyncForDevice(NvBufSurface* surf, int index, int plane) {
    (void)surf; (void)index; (void)plane;
    return 0;  /* no-op in mock */
}

static inline int NvBufSurfaceFromFd(int fd, void** surf_ptr) {
    (void)fd; (void)surf_ptr;
    return -1;  /* not implemented in mock */
}

static inline NvBufSurfTransform_Error
NvBufSurfTransform(NvBufSurface* src, NvBufSurface* dst,
                    NvBufSurfTransformParams* params) {
    if (!src || !dst || !params) return NvBufSurfTransformError_Invalid_Params;
    for (uint32_t i = 0; i < src->numFilled && i < dst->batchSize; i++) {
        uint32_t copy_size = src->surfaceList[i].dataSize;
        if (dst->surfaceList[i].dataSize < copy_size)
            copy_size = dst->surfaceList[i].dataSize;
        memcpy(dst->surfaceList[i].dataPtr, src->surfaceList[i].dataPtr, copy_size);
    }
    return NvBufSurfTransformError_Success;
}

#ifdef __cplusplus
}
#endif
