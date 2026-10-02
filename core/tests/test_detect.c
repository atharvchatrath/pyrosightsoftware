#include "ps_test.h"
#include "pyrosight/ps_detect.h"

#include <string.h>

static ps_thermal_frame_t f;
static uint16_t labels[PS_THERM_PIXELS];

static void rect(int x, int y, int w, int h, ps_dc_t v)
{
    for (int yy = y; yy < y + h; yy++)
        for (int xx = x; xx < x + w; xx++) f.px[yy * PS_THERM_W + xx] = v;
}

static void test_classical_finds_fire_and_person(void)
{
    ps_config_t cfg;
    ps_config_default(&cfg);
    for (int i = 0; i < PS_THERM_PIXELS; i++) f.px[i] = PS_DC(21);
    rect(10, 60, 20, 15, PS_DC(380)); /* fire */
    rect(100, 30, 12, 40, PS_DC(33)); /* standing person */
    rect(60, 5, 2, 2, PS_DC(200));    /* too small to be fire: a hot lamp */
    ps_detections_t d;
    ps_detect_classical(&cfg, &f, labels, &d);
    int fires = 0, people = 0;
    for (int i = 0; i < d.n; i++) {
        if (d.d[i].cls == PS_CLASS_FIRE) { fires++; CHECK_NEAR(d.d[i].x, 10, 0.1); CHECK_NEAR(d.d[i].w, 20, 0.1); }
        if (d.d[i].cls == PS_CLASS_PERSON) { people++; CHECK_NEAR(d.d[i].h, 40, 0.1); }
    }
    CHECK(fires == 1);
    CHECK(people == 1);
    CHECK(d.source == PS_DETECTOR_CLASSICAL);
}

static void test_warm_room_is_not_a_person(void)
{
    ps_config_t cfg;
    ps_config_default(&cfg);
    for (int i = 0; i < PS_THERM_PIXELS; i++) f.px[i] = PS_DC(34); /* the whole room is in the band */
    ps_detections_t d;
    ps_detect_classical(&cfg, &f, labels, &d);
    CHECK(d.n == 0);
}

static void test_iou_and_nms(void)
{
    ps_detections_t d;
    memset(&d, 0, sizeof d);
    d.n = 3;
    d.d[0] = (ps_detection_t){ .cls = PS_CLASS_PERSON, .score = 0.6f, .x = 10, .y = 10, .w = 10, .h = 30 };
    d.d[1] = (ps_detection_t){ .cls = PS_CLASS_PERSON, .score = 0.9f, .x = 11, .y = 11, .w = 10, .h = 30 };
    d.d[2] = (ps_detection_t){ .cls = PS_CLASS_FIRE, .score = 0.5f, .x = 11, .y = 11, .w = 10, .h = 30 };
    CHECK_NEAR(ps_iou(&d.d[0], &d.d[0]), 1.0, 1e-6);
    ps_nms(&d, 0.45f);
    CHECK(d.n == 2); /* overlapping fire survives: NMS is per class */
    CHECK_NEAR(d.d[0].score, 0.9, 1e-6);
}

static void test_centernet_decode(void)
{
    enum { GW = 40, GH = 30, P = GW * GH };
    static float heat[2 * P], wh[2 * P], off[2 * P];
    for (int i = 0; i < 2 * P; i++) heat[i] = -6.0f;
    memset(wh, 0, sizeof wh);
    memset(off, 0, sizeof off);
    /* Person centred at cell (20, 12) + offset (0.5, 0.25) -> pixel (82, 49). */
    int k = 12 * GW + 20;
    heat[P + k] = 2.0f;
    heat[P + k + 1] = 1.0f; /* neighbour, not a local max */
    wh[k] = 10; wh[P + k] = 30;
    off[k] = 0.5f; off[P + k] = 0.25f;
    ps_detections_t d;
    ps_centernet_decode(heat, wh, off, GW, GH, 4, 0.35f, &d);
    CHECK(d.n == 1);
    CHECK(d.d[0].cls == PS_CLASS_PERSON);
    CHECK_NEAR(d.d[0].x + d.d[0].w / 2, 82.0, 1e-3);
    CHECK_NEAR(d.d[0].y + d.d[0].h / 2, 49.0, 1e-3);
    CHECK_NEAR(d.d[0].score, 1.0 / (1.0 + exp(-2.0)), 1e-4);
}

static void test_distance(void)
{
    ps_config_t cfg;
    ps_config_default(&cfg);
    const double f_px = 80.0 / tan(28.5 * M_PI / 180.0);
    ps_detections_t d;
    memset(&d, 0, sizeof d);
    d.n = 2;
    double h5 = f_px * 1.7 / 5.0; /* standing person at 5 m */
    d.d[0] = (ps_detection_t){ .cls = PS_CLASS_PERSON, .x = 70, .y = 30, .w = 15, .h = (float)h5 };
    d.d[1] = (ps_detection_t){ .cls = PS_CLASS_PERSON, .x = 0, .y = 80, .w = 30, .h = 12 }; /* lying, at edge */
    ps_estimate_distances(&cfg, &d);
    CHECK_NEAR(d.d[0].dist_m, 5.0, 0.01);
    CHECK(!d.d[0].truncated);
    CHECK(d.d[0].dist_min_m < 5.0 && d.d[0].dist_max_m > 5.0);
    CHECK(d.d[1].truncated);
    CHECK_NEAR(d.d[1].dist_m, f_px * 1.7 / 30.0, 0.01);
}

int main(void)
{
    RUN(test_classical_finds_fire_and_person);
    RUN(test_warm_room_is_not_a_person);
    RUN(test_iou_and_nms);
    RUN(test_centernet_decode);
    RUN(test_distance);
    TEST_MAIN_END();
}
