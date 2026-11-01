/* CUDA Driver-API NF4 GEMV for Jetson Orin Nano Super (sm_87).
 * Links only against libcuda.so. PTX is loaded from nf4_gemv.ptx beside this .so.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0
#define CU_CTX_SCHED_AUTO 0

typedef CUresult (*fn_u32)(unsigned int);
typedef CUresult (*fn_dev)(CUdevice *, int);
typedef CUresult (*fn_ctx)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*fn_ctx_set)(CUcontext);
typedef CUresult (*fn_mod)(CUmodule *, const void *);
typedef CUresult (*fn_func)(CUfunction *, CUmodule, const char *);
typedef CUresult (*fn_alloc)(CUdeviceptr *, size_t);
typedef CUresult (*fn_free)(CUdeviceptr);
typedef CUresult (*fn_copy)(CUdeviceptr, const void *, size_t);
typedef CUresult (*fn_copy_back)(void *, CUdeviceptr, size_t);
typedef CUresult (*fn_launch)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void **, void **);
typedef CUresult (*fn_sync)(void);
typedef CUresult (*fn_errstr)(CUresult, const char **);

typedef struct {
    const void *host_key;
    CUdeviceptr device;
    size_t bytes;
} GpuBuf;

/* 42 layers × 7 mats × (packed+scales+code) needs >512 slots; was overflowing
 * and MiniTalk then permanently fell back to CPU (~1.4 tok/s). */
#define MAX_BUFS 2048
#define MAX_CACHE_BYTES (512ull * 1024ull * 1024ull)

static struct {
    void *lib;
    fn_u32 cuInit;
    fn_dev cuDeviceGet;
    fn_ctx cuCtxCreate;
    fn_ctx_set cuCtxSetCurrent;
    fn_mod cuModuleLoadData;
    fn_func cuModuleGetFunction;
    fn_alloc cuMemAlloc;
    fn_free cuMemFree;
    fn_copy cuMemcpyHtoD;
    fn_copy_back cuMemcpyDtoH;
    fn_launch cuLaunchKernel;
    fn_sync cuCtxSynchronize;
    fn_errstr cuGetErrorString;
    CUcontext ctx;
    CUmodule module;
    CUfunction kn;
    GpuBuf bufs[MAX_BUFS];
    int nbufs;
    size_t cache_bytes;
    CUdeviceptr scratch_x;
    CUdeviceptr scratch_y;
    size_t scratch_x_bytes;
    size_t scratch_y_bytes;
    int ready;
    char err[512];
} G;

static void set_err(CUresult rc, const char *what) {
    const char *msg = NULL;
    if (G.cuGetErrorString) {
        G.cuGetErrorString(rc, &msg);
    }
    snprintf(G.err, sizeof(G.err), "%s (%d) %s", what, (int)rc, msg ? msg : "");
}

static char *load_ptx_file(void) {
    Dl_info info;
    char path[PATH_MAX];
    FILE *fp;
    long sz;
    char *buf;
    /* Resolve directory of this shared library. */
    if (dladdr((void *)load_ptx_file, &info) && info.dli_fname) {
        snprintf(path, sizeof(path), "%s", info.dli_fname);
        char *slash = strrchr(path, '/');
        if (slash) {
            *slash = 0;
            strncat(path, "/nf4_gemv.ptx", sizeof(path) - strlen(path) - 1);
        } else {
            snprintf(path, sizeof(path), "nf4_gemv.ptx");
        }
    } else {
        snprintf(path, sizeof(path), "nf4_gemv.ptx");
    }
    fp = fopen(path, "rb");
    if (!fp) {
        snprintf(G.err, sizeof(G.err), "cannot open %s", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(fp);
        snprintf(G.err, sizeof(G.err), "empty PTX %s", path);
        return NULL;
    }
    buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(fp);
        snprintf(G.err, sizeof(G.err), "OOM reading PTX");
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        fclose(fp);
        free(buf);
        snprintf(G.err, sizeof(G.err), "short read %s", path);
        return NULL;
    }
    fclose(fp);
    buf[sz] = 0;
    return buf;
}

static int load_syms(void) {
    if (G.lib) {
        return 0;
    }
    G.lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!G.lib) {
        G.lib = dlopen("libcuda.so", RTLD_NOW);
    }
    if (!G.lib) {
        snprintf(G.err, sizeof(G.err), "dlopen(libcuda): %s", dlerror());
        return 1;
    }
#define S(n, t)                                                                \
    do {                                                                       \
        G.n = (t)dlsym(G.lib, #n);                                             \
        if (!G.n) {                                                            \
            snprintf(G.err, sizeof(G.err), "missing %s", #n);                  \
            return 1;                                                          \
        }                                                                      \
    } while (0)
    S(cuInit, fn_u32);
    S(cuDeviceGet, fn_dev);
    S(cuCtxCreate, fn_ctx);
    S(cuCtxSetCurrent, fn_ctx_set);
    S(cuModuleLoadData, fn_mod);
    S(cuModuleGetFunction, fn_func);
    S(cuMemAlloc, fn_alloc);
    S(cuMemFree, fn_free);
    S(cuMemcpyHtoD, fn_copy);
    S(cuMemcpyDtoH, fn_copy_back);
    S(cuLaunchKernel, fn_launch);
    S(cuCtxSynchronize, fn_sync);
    G.cuGetErrorString = (fn_errstr)dlsym(G.lib, "cuGetErrorString");
#undef S
    return 0;
}

static void cache_evict_one(void) {
    /* Drop oldest entry (FIFO). Enough to keep decode on-GPU under 8GB. */
    if (G.nbufs <= 0) {
        return;
    }
    G.cuMemFree(G.bufs[0].device);
    if (G.cache_bytes >= G.bufs[0].bytes) {
        G.cache_bytes -= G.bufs[0].bytes;
    } else {
        G.cache_bytes = 0;
    }
    memmove(&G.bufs[0], &G.bufs[1], (size_t)(G.nbufs - 1) * sizeof(GpuBuf));
    G.nbufs--;
}

static CUdeviceptr cache_upload(const void *host, size_t bytes) {
    int i;
    CUresult rc;
    CUdeviceptr d = 0;
    for (i = 0; i < G.nbufs; i++) {
        if (G.bufs[i].host_key == host && G.bufs[i].bytes == bytes) {
            /* Touch: move to end so hot weights survive eviction. */
            if (i != G.nbufs - 1) {
                GpuBuf tmp = G.bufs[i];
                memmove(&G.bufs[i], &G.bufs[i + 1], (size_t)(G.nbufs - 1 - i) * sizeof(GpuBuf));
                G.bufs[G.nbufs - 1] = tmp;
            }
            return G.bufs[G.nbufs - 1].device;
        }
    }
    while ((G.nbufs >= MAX_BUFS || G.cache_bytes + bytes > MAX_CACHE_BYTES) && G.nbufs > 0) {
        cache_evict_one();
    }
    rc = G.cuMemAlloc(&d, bytes);
    while (rc != CUDA_SUCCESS && G.nbufs > 0) {
        cache_evict_one();
        rc = G.cuMemAlloc(&d, bytes);
    }
    if (rc != CUDA_SUCCESS) {
        set_err(rc, "cuMemAlloc");
        return 0;
    }
    rc = G.cuMemcpyHtoD(d, host, bytes);
    if (rc != CUDA_SUCCESS) {
        set_err(rc, "cuMemcpyHtoD");
        G.cuMemFree(d);
        return 0;
    }
    G.bufs[G.nbufs].host_key = host;
    G.bufs[G.nbufs].device = d;
    G.bufs[G.nbufs].bytes = bytes;
    G.nbufs++;
    G.cache_bytes += bytes;
    return d;
}

static CUdeviceptr scratch_alloc(CUdeviceptr *slot, size_t *have, size_t need) {
    CUresult rc;
    if (*slot && *have >= need) {
        return *slot;
    }
    if (*slot) {
        G.cuMemFree(*slot);
        *slot = 0;
        *have = 0;
    }
    rc = G.cuMemAlloc(slot, need);
    if (rc != CUDA_SUCCESS) {
        set_err(rc, "cuMemAlloc scratch");
        *slot = 0;
        return 0;
    }
    *have = need;
    return *slot;
}

int cuda_nf4_init(void) {
    CUresult rc;
    CUdevice dev = 0;
    char *ptx;
    if (G.ready) {
        return 0;
    }
    if (load_syms()) {
        return 1;
    }
    rc = G.cuInit(0);
    if (rc != CUDA_SUCCESS) {
        set_err(rc, "cuInit");
        return 1;
    }
    rc = G.cuDeviceGet(&dev, 0);
    if (rc != CUDA_SUCCESS) {
        set_err(rc, "cuDeviceGet");
        return 1;
    }
    rc = G.cuCtxCreate(&G.ctx, CU_CTX_SCHED_AUTO, dev);
    if (rc != CUDA_SUCCESS) {
        set_err(rc, "cuCtxCreate");
        return 1;
    }
    ptx = load_ptx_file();
    if (!ptx) {
        return 1;
    }
    rc = G.cuModuleLoadData(&G.module, ptx);
    free(ptx);
    if (rc != CUDA_SUCCESS) {
        set_err(rc, "cuModuleLoadData");
        return 1;
    }
    rc = G.cuModuleGetFunction(&G.kn, G.module, "nf4_gemv");
    if (rc != CUDA_SUCCESS) {
        set_err(rc, "cuModuleGetFunction");
        return 1;
    }
    G.ready = 1;
    snprintf(G.err, sizeof(G.err), "Orin GPU ready (CUDA driver / sm_87)");
    return 0;
}

const char *cuda_nf4_last_error(void) { return G.err; }
int cuda_nf4_available(void) { return G.ready; }

int cuda_nf4_linear(const uint8_t *packed, const float *scales, const float *x, float *y,
                    int rows, int cols, int batch, const float *codebook) {
    CUdeviceptr d_packed, d_scales, d_code, d_x = 0, d_y = 0;
    CUresult rc;
    size_t packed_n, scale_n, x_n, y_n;
    void *args[7];

    if (!G.ready) {
        return 1;
    }
    if (batch != 1 || rows <= 0 || cols <= 0 || (cols % 64) != 0) {
        snprintf(G.err, sizeof(G.err), "shape not supported on GPU path");
        return 1;
    }
    G.cuCtxSetCurrent(G.ctx);
    packed_n = (size_t)rows * (size_t)(cols / 2);
    scale_n = (size_t)rows * (size_t)(cols / 64) * sizeof(float);
    x_n = (size_t)cols * sizeof(float);
    y_n = (size_t)rows * sizeof(float);

    d_packed = cache_upload(packed, packed_n);
    d_scales = cache_upload(scales, scale_n);
    d_code = cache_upload(codebook, 16 * sizeof(float));
    if (!d_packed || !d_scales || !d_code) {
        return 1;
    }
    d_x = scratch_alloc(&G.scratch_x, &G.scratch_x_bytes, x_n);
    d_y = scratch_alloc(&G.scratch_y, &G.scratch_y_bytes, y_n);
    if (!d_x || !d_y) {
        return 1;
    }
    if ((rc = G.cuMemcpyHtoD(d_x, x, x_n)) != CUDA_SUCCESS) {
        set_err(rc, "cuMemcpyHtoD x");
        return 1;
    }
    args[0] = &d_packed;
    args[1] = &d_scales;
    args[2] = &d_x;
    args[3] = &d_y;
    args[4] = &d_code;
    args[5] = &rows;
    args[6] = &cols;
    rc = G.cuLaunchKernel(G.kn, (unsigned)rows, 1, 1, 128, 1, 1, 0, NULL, args, NULL);
    if (rc != CUDA_SUCCESS) {
        set_err(rc, "cuLaunchKernel");
        return 1;
    }
    if ((rc = G.cuCtxSynchronize()) != CUDA_SUCCESS) {
        set_err(rc, "cuCtxSynchronize");
        return 1;
    }
    if ((rc = G.cuMemcpyDtoH(y, d_y, y_n)) != CUDA_SUCCESS) {
        set_err(rc, "cuMemcpyDtoH");
        return 1;
    }
    return 0;
}
