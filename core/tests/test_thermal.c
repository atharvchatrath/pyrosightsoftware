#include "ps_test.h"
#include "pyrosight/ps_thermal.h"

#include <string.h>

static ps_thermal_t th;
static ps_thermal_frame_t f;
static ps_dc_t scratch[PS_THERM_PIXELS];

static void fill(ps_dc_t v, float noise_dc)
{
    for (int i = 0; i < PS_THERM_PIXELS; i++) f.px[i] = (ps_dc_t)(v + noise_dc * ps_test_gauss());
}

static double stddev(void)
{
    double m = 0, s = 0;
    for (int i = 0; i < PS_THERM_PIXELS; i++) m += f.px[i];
    m /= PS_THERM_PIXELS;
    for (int i = 0; i < PS_THERM_PIXELS; i++) s += (f.px[i] - m) * (f.px[i] - m);
    return sqrt(s / PS_THERM_PIXELS);
}

static void test_model_code(void)
{
    CHECK(ps_thermal_model_code(PS_DC(-40)) == 0);
    CHECK(ps_thermal_model_code(PS_DC(-20)) == 0);
    CHECK(ps_thermal_model_code(PS_DC(60)) == 192);
    CHECK(ps_thermal_model_code(PS_DC(600)) == 255);
    CHECK(ps_thermal_model_code(PS_DC(1000)) == 255);
    /* Monotonic, and body temperature resolves from room temperature. */
    int prev = -1;
    for (int dc = -300; dc <= 7000; dc += 7) {
        int c = ps_thermal_model_code((ps_dc_t)dc);
        CHECK(c >= prev);
        prev = c;
    }
    CHECK(ps_thermal_model_code(PS_DC(35)) - ps_thermal_model_code(PS_DC(22)) >= 25);
}

static void test_denoise_reduces_noise(void)
{
    ps_config_t cfg;
    ps_config_default(&cfg);
    ps_thermal_init(&th);
    double raw_sd = 0, out_sd = 0;
    for (int k = 0; k < 12; k++) {
        fill(PS_DC(25), 5.0f); /* 0.5 C noise */
        raw_sd = stddev();
        ps_thermal_denoise(&th, &cfg, &f, scratch);
        out_sd = stddev();
    }
    printf("  noise: raw %.2f dC -> denoised %.2f dC\n", raw_sd, out_sd);
    CHECK(out_sd < raw_sd * 0.5);
}

static void test_median_removes_hot_pixel(void)
{
    ps_config_t cfg;
    ps_config_default(&cfg);
    cfg.temporal_shift = 0;
    ps_thermal_init(&th);
    fill(PS_DC(20), 0);
    f.px[60 * PS_THERM_W + 80] = PS_DC(300);
    ps_thermal_denoise(&th, &cfg, &f, scratch);
    CHECK(f.px[60 * PS_THERM_W + 80] == PS_DC(20));
    CHECK(th.max_dc == PS_DC(20));
}

static void test_motion_bypasses_iir(void)
{
    ps_config_t cfg;
    ps_config_default(&cfg);
    cfg.median3x3 = false;
    ps_thermal_init(&th);
    fill(PS_DC(20), 0);
    ps_thermal_denoise(&th, &cfg, &f, scratch);
    fill(PS_DC(20), 0);
    f.px[1000] = PS_DC(36); /* person walks in */
    ps_thermal_denoise(&th, &cfg, &f, scratch);
    CHECK(f.px[1000] == PS_DC(36)); /* no ghosting/lag on a real change */
}

static void test_display_mapping(void)
{
    ps_config_t cfg;
    ps_config_default(&cfg);
    cfg.detail_gain_q4 = 0;
    ps_thermal_init(&th);
    static ps_gray_frame_t g;
    for (int y = 0; y < PS_THERM_H; y++)
        for (int x = 0; x < PS_THERM_W; x++) f.px[y * PS_THERM_W + x] = (ps_dc_t)(PS_DC(15) + x * 2);
    ps_thermal_to_display(&th, &cfg, &f, &g);
    CHECK(g.px[60 * PS_THERM_W + 0] < 10);
    CHECK(g.px[60 * PS_THERM_W + 159] > 245);
    CHECK(g.px[60 * PS_THERM_W + 40] < g.px[60 * PS_THERM_W + 120]);

    /* A fire in view must not crush the room to black. */
    ps_thermal_init(&th);
    fill(PS_DC(22), 0);
    for (int y = 0; y < PS_THERM_H; y++) f.px[y * PS_THERM_W + 10] = PS_DC(28); /* door frame */
    for (int i = 0; i < 400; i++) f.px[i] = PS_DC(450);
    ps_thermal_to_display(&th, &cfg, &f, &g);
    CHECK(g.px[60 * PS_THERM_W + 10] - g.px[60 * PS_THERM_W + 50] > 20);
}

int main(void)
{
    RUN(test_model_code);
    RUN(test_denoise_reduces_noise);
    RUN(test_median_removes_hot_pixel);
    RUN(test_motion_bypasses_iir);
    RUN(test_display_mapping);
    TEST_MAIN_END();
}
