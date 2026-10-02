/* Host tests: debounced short/long press classification. */
#include "../../core/tests/ps_test.h"
#include "button_fsm.h"

static btn_fsm_t B;
static int shorts, longs;
static uint32_t t_long;

/* Drive the FSM at 5 ms steps with a level function. */
static void run(uint32_t t0, uint32_t t1, bool level)
{
    for (uint32_t t = t0; t < t1; t += 5) {
        btn_event_t e = btn_fsm_update(&B, level, t);
        if (e == BTN_EV_SHORT) shorts++;
        if (e == BTN_EV_LONG) { longs++; t_long = t; }
    }
}

static void reset(void) { btn_fsm_init(&B, 30, 1500); shorts = longs = 0; run(0, 100, false); }

static void test_short(void)
{
    reset();
    run(100, 300, true);
    run(300, 400, false);
    CHECK(shorts == 1 && longs == 0);
}

static void test_long(void)
{
    reset();
    run(100, 2000, true);
    CHECK(longs == 1 && shorts == 0);
    CHECK(t_long >= 1600 && t_long <= 1610);   /* 1.5 s after the first edge (+ step) */
    run(2000, 2100, false);
    CHECK(longs == 1 && shorts == 0);          /* release after long is silent */
    run(2100, 5000, true);                     /* held very long: one long only */
    CHECK(longs == 2);
}

static void test_bounce(void)
{
    reset();
    /* Contact bounce on press and release: 10 ms toggles. */
    uint32_t t = 100;
    for (int i = 0; i < 6; i++, t += 10) { run(t, t + 5, i % 2 == 0); run(t + 5, t + 10, i % 2 != 0); }
    run(t, t + 200, true);
    t += 200;
    for (int i = 0; i < 6; i++, t += 10) { run(t, t + 5, false); run(t + 5, t + 10, true); }
    run(t, t + 200, false);
    CHECK(shorts == 1 && longs == 0);
    /* A glitch shorter than the debounce time is ignored. */
    reset();
    run(100, 120, true);
    run(120, 400, false);
    CHECK(shorts == 0 && longs == 0);
}

static void test_held_at_boot(void)
{
    btn_fsm_init(&B, 30, 1500);
    shorts = longs = 0;
    run(0, 3000, true);
    run(3000, 3200, false);
    CHECK(shorts == 0 && longs == 0);
    run(3200, 3400, true);
    run(3400, 3600, false);
    CHECK(shorts == 1);
}

static void test_wraparound(void)
{
    btn_fsm_init(&B, 30, 1500);
    shorts = longs = 0;
    uint32_t t0 = 0xFFFFFF00u;
    for (uint32_t i = 0; i < 100; i += 5) btn_fsm_update(&B, false, t0 + i);
    for (uint32_t i = 100; i < 1800; i += 5)
        if (btn_fsm_update(&B, true, t0 + i) == BTN_EV_LONG) longs++;
    CHECK(longs == 1);
}

int main(void)
{
    RUN(test_short);
    RUN(test_long);
    RUN(test_bounce);
    RUN(test_held_at_boot);
    RUN(test_wraparound);
    TEST_MAIN_END();
}
