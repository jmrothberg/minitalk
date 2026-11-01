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

#define MAX_BUFS 512

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

static CUdeviceptr cache_upload(const void *host, size_t bytes) {
    int i;
    CUresult rc;
    CUdeviceptr d = 0;
    for (i = 0; i < G.nbufs; i++) {
        if (G.bufs[i].host_key == host && G.bufs[i].bytes == bytes) {
            return G.bufs[i].device;
        }
    }
    if (G.nbufs >= MAX_BUFS) {
        snprintf(G.err, sizeof(G.err), "gpu buffer cache full");
        return 0;
    }
    rc = G.cuMemAlloc(&d, bytes);
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
    return d;
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
    if ((rc = G.cuMemAlloc(&d_x, x_n)) != CUDA_SUCCESS ||
        (rc = G.cuMemAlloc(&d_y, y_n)) != CUDA_SUCCESS) {
        set_err(rc, "cuMemAlloc x/y");
        goto fail;
    }
    if ((rc = G.cuMemcpyHtoD(d_x, x, x_n)) != CUDA_SUCCESS) {
        set_err(rc, "cuMemcpyHtoD x");
        goto fail;
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
        goto fail;
    }
    if ((rc = G.cuCtxSynchronize()) != CUDA_SUCCESS) {
        set_err(rc, "cuCtxSynchronize");
        goto fail;
    }
    if ((rc = G.cuMemcpyDtoH(y, d_y, y_n)) != CUDA_SUCCESS) {
        set_err(rc, "cuMemcpyDtoH");
        goto fail;
    }
    G.cuMemFree(d_x);
    G.cuMemFree(d_y);
    return 0;
fail:
    if (d_x)
        G.cuMemFree(d_x);
    if (d_y)
        G.cuMemFree(d_y);
    return 1;
}
