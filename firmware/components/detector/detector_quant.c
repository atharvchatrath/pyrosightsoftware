#include "detector_quant.h"

#include <math.h>

void dq_build_input_lut(int exponent, int8_t lut[256])
{
    const float scale = ldexpf(1.0f, -exponent) / 255.0f;
    for (int c = 0; c < 256; c++) {
        float q = rintf((float)c * scale);   /* default rounding mode: half to even, as ESP-DL */
        if (q > 127.0f) q = 127.0f;
        if (q < -128.0f) q = -128.0f;
        lut[c] = (int8_t)q;
    }
}

void dq_quantize_input(const uint8_t *codes, int n, const int8_t lut[256], int8_t *out)
{
    for (int i = 0; i < n; i++) out[i] = lut[codes[i]];
}

dq_layout_t dq_layout(const int *s, int ndim, int c, int h, int w)
{
    if (ndim == 4 && s[0] == 1) {
        if (s[1] == h && s[2] == w && s[3] == c) return DQ_LAYOUT_NHWC;
        if (s[1] == c && s[2] == h && s[3] == w) return DQ_LAYOUT_NCHW;
    } else if (ndim == 3) {
        if (s[0] == h && s[1] == w && s[2] == c) return DQ_LAYOUT_NHWC;
        if (s[0] == c && s[1] == h && s[2] == w) return DQ_LAYOUT_NCHW;
    }
    return DQ_LAYOUT_UNKNOWN;
}

#define DEQUANT_BODY                                                         \
    const float sc = ldexpf(1.0f, exponent);                                 \
    const int plane = h * w;                                                 \
    if (layout == DQ_LAYOUT_NCHW) {                                          \
        for (int i = 0; i < c * plane; i++) dst[i] = src[i] * sc;            \
    } else {                                                                 \
        for (int p = 0; p < plane; p++)                                      \
            for (int k = 0; k < c; k++) dst[k * plane + p] = src[p * c + k] * sc; \
    }

void dq_dequant_i8_to_chw(const int8_t *src, dq_layout_t layout, int c, int h, int w, int exponent, float *dst)
{
    DEQUANT_BODY
}

void dq_dequant_i16_to_chw(const int16_t *src, dq_layout_t layout, int c, int h, int w, int exponent, float *dst)
{
    DEQUANT_BODY
}
