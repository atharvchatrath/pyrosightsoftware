/* Integration: the whole core driven the way the firmware drives it. */
#include "ps_test.h"
#include "pyrosight/ps_system.h"

#include <string.h>

static ps_system_t sys;
static ps_thermal_frame_t raw;
static ps_fb_t fb;

static void make_frame(uint32_t id, uint32_t t)
{
    for (int i = 0; i < PS_THERM_PIXELS; i++) raw.px[i] = (ps_dc_t)(PS_DC(22) + 3 * ps_test_gauss());
    for (int y = 30; y < 75; y++)
        for (int x = 100; x < 114; x++) raw.px[y * PS_THERM_W + x] = PS_DC(33);
    for (int y = 80; y < 100; y++)
        for (int x = 10; x < 40; x++) raw.px[y * PS_THERM_W + x] = PS_DC(420);
    raw.frame_id = id;
    raw.t_ms = t;
}

static int count_color(uint16_t c)
{
    int n = 0;
    for (int i = 0; i < PS_DISP_W * PS_DISP_H; i++) n += fb.px[i] == c;
    return n;
}

static void test_pipeline(void)
{
    ps_system_init(&sys, NULL, 0);
    uint32_t t = 0, id = 0;
    ps_system_on_yaw(&sys, 0.3f, t);
    ps_system_on_button(&sys, PS_BTN_MARK_ENTRY, t);
    for (int k = 0; k < 100; k++) {
        t += 115;
        make_frame(++id, t);
        ps_system_on_frame(&sys, &raw);
        ps_system_run_classical(&sys);
        ps_system_on_yaw(&sys, 0.3f, t);
        ps_system_tick(&sys, t);
    }
    CHECK(sys.camera_ok);
    CHECK(sys.stats.frames == 100 && sys.stats.frames_dropped == 0);
    int fires = 0, people = 0;
    for (int i = 0; i < sys.dets.n; i++) {
        fires += sys.dets.d[i].cls == PS_CLASS_FIRE;
        people += sys.dets.d[i].cls == PS_CLASS_PERSON;
    }
    CHECK(fires == 1 && people == 1);

    ps_system_render(&sys, &fb, t);
    CHECK(count_color(ps_rgb565(255, 90, 0)) > 1000);   /* fire overlay */
    CHECK(count_color(PS_COLOR_PERSON) > 50);            /* person box (white) */
    CHECK(count_color(PS_COLOR_FIRE) > 50);              /* fire box (purple) */
    CHECK(count_color(ps_rgb565(40, 255, 80)) > 30);    /* green nav ring/arrow */

    ps_alert_t a;
    bool entry = false, person = false;
    while (ps_system_next_alert(&sys, &a)) {
        entry |= a.parts[0] == PS_PHRASE_ENTRY_MARKED;
        person |= a.parts[0] == PS_PHRASE_PERSON;
    }
    CHECK(entry && !person); /* classical detections are shown, not spoken */
    ps_detections_t nd = sys.dets;
    nd.source = PS_DETECTOR_NEURAL;
    ps_system_on_detections(&sys, &nd, t);
    person = false;
    while (ps_system_next_alert(&sys, &a)) person |= a.parts[0] == PS_PHRASE_PERSON;
    CHECK(person);
}

static void test_camera_and_imu_loss(void)
{
    uint32_t t = sys.frame.t_ms;
    for (int k = 0; k < 30; k++) { t += 50; ps_system_tick(&sys, t); }
    CHECK(!sys.camera_ok);
    CHECK(!sys.imu_ok);
    ps_alert_t a;
    bool cam = false, imu = false;
    while (ps_system_next_alert(&sys, &a)) {
        cam |= a.parts[0] == PS_PHRASE_CAMERA_LOST;
        imu |= a.parts[0] == PS_PHRASE_MOTION_LOST;
    }
    CHECK(cam && imu);
    CHECK(sys.dets.n == 0); /* stale boxes cleared */
    ps_system_render(&sys, &fb, 0);
    CHECK(count_color(ps_rgb565(255, 40, 40)) > 100); /* THERMAL LOST banner */

    make_frame(sys.frame.frame_id + 5, t + 10);
    ps_system_on_frame(&sys, &raw);
    CHECK(sys.camera_ok);
    CHECK(sys.stats.frames_dropped == 4);
}

static void test_buttons(void)
{
    ps_palette_t p = sys.disp.palette;
    ps_system_on_button(&sys, PS_BTN_PALETTE, 0);
    CHECK(sys.disp.palette != p);
    ps_system_on_battery(&sys, 3300, 0);
    CHECK(ps_system_brightness(&sys) <= sys.policy.max_brightness);
}

int main(void)
{
    RUN(test_pipeline);
    RUN(test_camera_and_imu_loss);
    RUN(test_buttons);
    TEST_MAIN_END();
}
