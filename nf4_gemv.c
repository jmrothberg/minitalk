/* Fused NF4 and float16 matrix-vector kernels for MiniTalk.
 * Weights stay packed; each row is unpacked into a 64-wide tile and dotted
 * with NEON so we never materialize a full float matrix in RAM.
 */
#include <arm_neon.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { NF4_BLOCK = 64 };

static inline float dot64(const float *w, const float *x) {
    float32x4_t a0 = vdupq_n_f32(0.f);
    float32x4_t a1 = vdupq_n_f32(0.f);
    float32x4_t a2 = vdupq_n_f32(0.f);
    float32x4_t a3 = vdupq_n_f32(0.f);
    for (int k = 0; k < 64; k += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(w + k), vld1q_f32(x + k));
        a1 = vfmaq_f32(a1, vld1q_f32(w + k + 4), vld1q_f32(x + k + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(w + k + 8), vld1q_f32(x + k + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(w + k + 12), vld1q_f32(x + k + 12));
    }
    a0 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    return vaddvq_f32(a0);
}

static inline void unpack_block(const uint8_t *bytes, float scale, const float *code, float *w) {
    for (int i = 0; i < 32; i++) {
        uint8_t byte = bytes[i];
        /* bitsandbytes stores the first weight in the high nibble. */
        w[2 * i] = code[byte >> 4] * scale;
        w[2 * i + 1] = code[byte & 0x0f] * scale;
    }
}

/* Y[b, r] = dot(X[b], dequant(W[r])). x is [batch, cols], y is [batch, rows]. */
int nf4_linear(const uint8_t *packed, const float *absmax, const float *x, float *y,
               int rows, int cols, int batch, const float *codebook) {
    if (cols <= 0 || (cols % NF4_BLOCK) != 0 || batch <= 0 || rows <= 0) {
        return 1;
    }
    const int blocks = cols / NF4_BLOCK;

#pragma omp parallel
    {
        /* One accumulator per token so each weight row is read a single time. */
        float *acc = (float *)malloc((size_t)batch * sizeof(float));
        if (acc != NULL) {
#pragma omp for schedule(static)
            for (int r = 0; r < rows; r++) {
                memset(acc, 0, (size_t)batch * sizeof(float));
                const uint8_t *wp = packed + (size_t)r * (size_t)(cols >> 1);
                const float *row_scale = absmax + (size_t)r * (size_t)blocks;
                for (int blk = 0; blk < blocks; blk++) {
                    float w[NF4_BLOCK];
                    unpack_block(wp, row_scale[blk], codebook, w);
                    wp += 32;
                    const int col = blk * NF4_BLOCK;
                    for (int b = 0; b < batch; b++) {
                        const float *xv = x + ((size_t)b * (size_t)cols) + col;
                        acc[b] += dot64(w, xv);
                    }
                }
                for (int b = 0; b < batch; b++) {
                    y[((size_t)b * (size_t)rows) + r] = acc[b];
                }
            }
        }
        free(acc);
    }
    return 0;
}

static inline float dot_f16(const uint16_t *w, const float *x, int cols) {
    float32x4_t a0 = vdupq_n_f32(0.f);
    float32x4_t a1 = vdupq_n_f32(0.f);
    int k = 0;
    for (; k + 16 <= cols; k += 16) {
        float16x8_t h0 = vld1q_f16((const __fp16 *)(w + k));
        float16x8_t h1 = vld1q_f16((const __fp16 *)(w + k + 8));
        a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(h0)), vld1q_f32(x + k));
        a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_high_f16(h0)), vld1q_f32(x + k + 4));
        a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(h1)), vld1q_f32(x + k + 8));
        a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_high_f16(h1)), vld1q_f32(x + k + 12));
    }
    float sum = vaddvq_f32(vaddq_f32(a0, a1));
    for (; k < cols; k++) {
        sum += (float)((const __fp16 *)w)[k] * x[k];
    }
    return sum;
}

/* W is float16 [rows, cols] row-major. Used for the unquantized lm_head. */
int f16_linear(const uint16_t *w, const float *x, float *y, int rows, int cols, int batch) {
    if (rows <= 0 || cols <= 0 || batch <= 0) {
        return 1;
    }
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const uint16_t *row = w + (size_t)r * (size_t)cols;
        for (int b = 0; b < batch; b++) {
            y[((size_t)b * (size_t)rows) + r] = dot_f16(row, x + (size_t)b * (size_t)cols, cols);
        }
    }
    return 0;
}
