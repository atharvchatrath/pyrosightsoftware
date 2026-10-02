/*
 * ESP-DL tensor quantisation helpers for the PyroSight detector (hardware
 * independent, host-tested).
 *
 * ESP-DL uses power-of-two scales: real = q * 2^exponent. The model was
 * trained on x = code / 255 (code = ps_system_t.model_input, uint8), so the
 * int8 input is
 *     q = clamp(round_half_even(code / 255 * 2^-exponent), -128, 127)
 * with the input tensor's exponent (typically -7, i.e. q = code * 128/255;
 * code 255 saturates at 127, an error of < 0.8 % at full scale).
 * Outputs are dequantised with their own exponents and re-ordered from the
 * tensor layout (ESP-DL is NHWC: [1, H, W, C]) to the channel-major planes
 * that ps_centernet_decode() expects.
 */
#ifndef DETECTOR_QUANT_H
#define DETECTOR_QUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DQ_LAYOUT_UNKNOWN = 0,
    DQ_LAYOUT_NHWC,
    DQ_LAYOUT_NCHW,
} dq_layout_t;

/* 256-entry LUT code -> int8 for an input exponent. */
void dq_build_input_lut(int exponent, int8_t lut[256]);
void dq_quantize_input(const uint8_t *codes, int n, const int8_t lut[256], int8_t *out);

/* Decide the layout of a 4-D shape for an expected C x H x W tensor. */
dq_layout_t dq_layout(const int *shape, int ndim, int c, int h, int w);

/* int8 / int16 tensor -> float [C][H][W]. */
void dq_dequant_i8_to_chw(const int8_t *src, dq_layout_t layout, int c, int h, int w, int exponent, float *dst);
void dq_dequant_i16_to_chw(const int16_t *src, dq_layout_t layout, int c, int h, int w, int exponent, float *dst);

#ifdef __cplusplus
}
#endif

#endif
