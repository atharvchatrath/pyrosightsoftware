/* Host tests: detector input quantisation, output dequantisation/re-layout,
 * and the full path into the core's ps_centernet_decode(). */
#include <string.h>

#include "../../core/tests/ps_test.h"
#include "detector_quant.h"
#include "pyrosight/ps_detect.h"

#define GW 40
#define GH 30
#define PL (GW * GH)

static void test_input(void)
{
    int8_t lut[256];
    dq_build_input_lut(-7, lut);
    CHECK(lut[0] == 0);
    CHECK(lut[255] == 127);           /* 128 saturates */
    CHECK(lut[128] == 64);            /* 128*128/255 = 64.25 */
    CHECK(lut[2] == 1);               /* 1.004 */
    for (int c = 0; c < 255; c++) CHECK(lut[c] <= lut[c + 1]);
    for (int c = 0; c < 255; c++) CHECK_NEAR(lut[c] / 128.0, c / 255.0, 0.5 / 128 + 1e-6);
    dq_build_input_lut(-8, lut);      /* exponent -8: 1.0 -> 256 saturates at 127 */
    CHECK(lut[255] == 127 && lut[127] == 127 && lut[63] == 63);
    /* Round half to even: code 51 at exponent -8 -> 51.2 -> 51. */
    dq_build_input_lut(-6, lut);      /* code*64/255; code 255 -> 64 */
    CHECK(lut[255] == 64);
    uint8_t codes[4] = { 0, 51, 204, 255 };
    int8_t q[4];
    dq_quantize_input(codes, 4, lut, q);
    CHECK(q[0] == 0 && q[1] == 13 && q[2] == 51 && q[3] == 64);   /* 12.8->13, 51.2->51 */
}

static void test_layout(void)
{
    int nhwc[4] = { 1, GH, GW, 2 }, nchw[4] = { 1, 2, GH, GW }, bad[4] = { 1, 3, GH, GW };
    CHECK(dq_layout(nhwc, 4, 2, GH, GW) == DQ_LAYOUT_NHWC);
    CHECK(dq_layout(nchw, 4, 2, GH, GW) == DQ_LAYOUT_NCHW);
    CHECK(dq_layout(bad, 4, 2, GH, GW) == DQ_LAYOUT_UNKNOWN);
    int hwc[3] = { GH, GW, 2 };
    CHECK(dq_layout(hwc, 3, 2, GH, GW) == DQ_LAYOUT_NHWC);

    static int8_t src[2 * PL];
    static float dst[2 * PL];
    for (int p = 0; p < PL; p++) { src[p * 2] = (int8_t)(p % 100); src[p * 2 + 1] = (int8_t)(-(p % 50)); }
    dq_dequant_i8_to_chw(src, DQ_LAYOUT_NHWC, 2, GH, GW, -2, dst);
    CHECK_NEAR(dst[0 * PL + 123], 23 * 0.25, 1e-6);
    CHECK_NEAR(dst[1 * PL + 123], -23 * 0.25, 1e-6);
    dq_dequant_i8_to_chw(src, DQ_LAYOUT_NCHW, 2, GH, GW, 1, dst);
    CHECK_NEAR(dst[5], src[5] * 2.0, 1e-6);
    static int16_t s16[2 * PL];
    for (int i = 0; i < 2 * PL; i++) s16[i] = (int16_t)(i - 1000);
    dq_dequant_i16_to_chw(s16, DQ_LAYOUT_NHWC, 2, GH, GW, -4, dst);
    CHECK_NEAR(dst[PL + 10], (21 - 1000) / 16.0, 1e-6);
}

/* Synthetic NHWC model outputs with one person at grid (gx=20, gy=10). */
static void test_end_to_end(void)
{
    static int8_t heat[PL * 2], wh[PL * 2], off[PL * 2];
    const int he = -3, we = 0, oe = -7;           /* exponents */
    memset(heat, (int8_t)-64, sizeof(heat));      /* -8.0 logits everywhere */
    memset(wh, 0, sizeof(wh));
    memset(off, 0, sizeof(off));
    const int gx = 20, gy = 10, p = gy * GW + gx;
    heat[p * 2 + 1] = 24;                         /* person logit 3.0 -> 0.95 */
    heat[(p + 1) * 2 + 1] = 8;                    /* weaker neighbour, suppressed */
    wh[p * 2 + 0] = 12; wh[p * 2 + 1] = 40;       /* 12 x 40 px */
    off[p * 2 + 0] = 64; off[p * 2 + 1] = 32;     /* 0.5, 0.25 */

    static float fh[2 * PL], fw[2 * PL], fo[2 * PL];
    dq_dequant_i8_to_chw(heat, DQ_LAYOUT_NHWC, 2, GH, GW, he, fh);
    dq_dequant_i8_to_chw(wh, DQ_LAYOUT_NHWC, 2, GH, GW, we, fw);
    dq_dequant_i8_to_chw(off, DQ_LAYOUT_NHWC, 2, GH, GW, oe, fo);
    ps_detections_t d;
    memset(&d, 0, sizeof(d));
    ps_centernet_decode(fh, fw, fo, GW, GH, 4, 0.35f, &d);
    CHECK(d.n == 1);
    CHECK(d.source == PS_DETECTOR_NEURAL);
    CHECK(d.d[0].cls == PS_CLASS_PERSON);
    CHECK_NEAR(d.d[0].score, 0.9526, 1e-3);
    CHECK_NEAR(d.d[0].x + d.d[0].w / 2, (gx + 0.5) * 4, 1e-4);
    CHECK_NEAR(d.d[0].y + d.d[0].h / 2, (gy + 0.25) * 4, 1e-4);
    CHECK_NEAR(d.d[0].w, 12, 1e-4);
    CHECK_NEAR(d.d[0].h, 40, 1e-4);
}

int main(void)
{
    RUN(test_input);
    RUN(test_layout);
    RUN(test_end_to_end);
    TEST_MAIN_END();
}
