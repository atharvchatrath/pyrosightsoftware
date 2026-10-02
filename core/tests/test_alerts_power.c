#include "ps_test.h"
#include "pyrosight/ps_alerts.h"
#include "pyrosight/ps_power.h"

#include <string.h>

static void test_direction_phrases(void)
{
    CHECK(ps_direction_phrase(0) == PS_PHRASE_DIR_AHEAD);
    CHECK(ps_direction_phrase(20) == PS_PHRASE_DIR_AHEAD);
    CHECK(ps_direction_phrase(45) == PS_PHRASE_DIR_AHEAD_LEFT);
    CHECK(ps_direction_phrase(90) == PS_PHRASE_DIR_LEFT);
    CHECK(ps_direction_phrase(-90) == PS_PHRASE_DIR_RIGHT);
    CHECK(ps_direction_phrase(179) == PS_PHRASE_DIR_BEHIND);
    CHECK(ps_direction_phrase(-179) == PS_PHRASE_DIR_BEHIND);
    CHECK(ps_direction_phrase(-140) == PS_PHRASE_DIR_BEHIND_RIGHT);
    for (int p = 1; p < PS_PHRASE_COUNT; p++) CHECK(ps_phrase_text[p] && ps_phrase_text[p][0]);
}

static void test_queue_priority(void)
{
    ps_alerts_t a;
    ps_alerts_init(&a);
    ps_alerts_push(&a, PS_PRIO_INFO, 1, PS_PHRASE_ENTRY_MARKED, 0, 0);
    ps_alerts_push(&a, PS_PRIO_CRITICAL, 2, PS_PHRASE_CAMERA_LOST, 0, 0);
    ps_alerts_push(&a, PS_PRIO_CRITICAL, 3, PS_PHRASE_CAMERA_LOST, 0, 0); /* duplicate: merged */
    ps_alert_t out;
    CHECK(ps_alerts_peek_max_prio(&a) == PS_PRIO_CRITICAL);
    CHECK(ps_alerts_pop(&a, &out) && out.parts[0] == PS_PHRASE_CAMERA_LOST);
    CHECK(ps_alerts_pop(&a, &out) && out.parts[0] == PS_PHRASE_ENTRY_MARKED);
    CHECK(!ps_alerts_pop(&a, &out));
    /* Full queue: a critical alert evicts an info one. */
    for (int i = 0; i < PS_ALERT_QUEUE; i++) ps_alerts_push(&a, PS_PRIO_INFO, (uint32_t)i, (ps_phrase_t)(PS_PHRASE_DIR_AHEAD + i), 0, 0);
    ps_alerts_push(&a, PS_PRIO_CRITICAL, 99, PS_PHRASE_BATTERY_CRITICAL, 0, 0);
    CHECK(ps_alerts_pop(&a, &out) && out.parts[0] == PS_PHRASE_BATTERY_CRITICAL);
}

static int count_phrase(ps_alerts_t *a, ps_phrase_t p)
{
    int n = 0;
    ps_alert_t out;
    while (ps_alerts_pop(a, &out)) if (out.parts[0] == p) n++;
    return n;
}

static void test_nav_alert_levels(void)
{
    ps_config_t cfg;
    ps_config_default(&cfg);
    ps_alerts_t a;
    ps_alerts_init(&a);
    ps_nav_guidance_t g = { .valid = true, .route_bearing_rel_deg = 170, .home_dist_m = 20, .route_dist_m = 25, .confidence = 0.9f };
    uint32_t t = 0;
    for (; t < 30000; t += 100) ps_alerts_update_nav(&a, &cfg, &g, t);
    CHECK(a.n == 0); /* confident: silent */

    g.confidence = 0.5f;
    ps_alerts_update_nav(&a, &cfg, &g, t);
    ps_alert_t out;
    CHECK(ps_alerts_pop(&a, &out));
    CHECK(out.parts[0] == PS_PHRASE_WAY_OUT_IS && out.parts[1] == PS_PHRASE_DIR_BEHIND);
    /* Repeats every direction_repeat_ms while degraded. */
    for (uint32_t k = 0; k < 31000; k += 100) ps_alerts_update_nav(&a, &cfg, &g, t + k);
    CHECK(count_phrase(&a, PS_PHRASE_WAY_OUT_IS) == 1); /* identical waiting alerts merge */

    /* Chatter around the warn threshold must not re-trigger. */
    ps_alerts_init(&a);
    a.level = PS_NAVCONF_DEGRADED;
    g.confidence = 0.62f; /* above warn, below warn + hysteresis */
    ps_alerts_update_nav(&a, &cfg, &g, 1);
    CHECK(a.level == PS_NAVCONF_DEGRADED);

    g.confidence = 0.2f;
    ps_alerts_update_nav(&a, &cfg, &g, 2);
    CHECK(a.level == PS_NAVCONF_UNRELIABLE);
    CHECK(ps_alerts_pop(&a, &out) && out.parts[0] == PS_PHRASE_NAV_UNRELIABLE && out.parts[1] == PS_PHRASE_FOLLOW_HOSE);
}

static void test_person_alert(void)
{
    ps_config_t cfg;
    ps_config_default(&cfg);
    ps_alerts_t a;
    ps_alerts_init(&a);
    ps_detections_t d;
    memset(&d, 0, sizeof d);
    d.n = 1;
    d.d[0] = (ps_detection_t){ .cls = PS_CLASS_PERSON, .score = 0.8f, .x = 10, .y = 40, .w = 12, .h = 40 };
    d.source = PS_DETECTOR_CLASSICAL;
    ps_alerts_update_detections(&a, &cfg, &d, 500);
    CHECK(a.n == 0); /* threshold fallback never speaks */
    d.source = PS_DETECTOR_NEURAL;
    ps_alerts_update_detections(&a, &cfg, &d, 500);
    ps_alert_t out;
    CHECK(ps_alerts_pop(&a, &out) && out.parts[0] == PS_PHRASE_PERSON && out.parts[1] == PS_PHRASE_DIR_AHEAD_LEFT);
    for (uint32_t t = 600; t < 30000; t += 115) ps_alerts_update_detections(&a, &cfg, &d, t);
    CHECK(a.n == 0); /* continuously visible: announced once */
}

static void test_power(void)
{
    ps_config_t cfg;
    ps_config_default(&cfg);
    CHECK(ps_battery_pct(4200) == 100);
    CHECK(ps_battery_pct(3000) == 0);
    CHECK(ps_battery_pct(3700) == 20);
    ps_power_t p;
    ps_power_init(&p);
    ps_power_policy_t pol;
    CHECK(!ps_power_update(&p, &cfg, 3900));
    CHECK(p.mode == PS_PWR_NORMAL);
    bool changed = false;
    for (int i = 0; i < 60; i++) changed |= ps_power_update(&p, &cfg, 3450);
    CHECK(changed && p.mode == PS_PWR_ECO);
    ps_power_policy(&p, &pol);
    CHECK(pol.max_infer_fps == 8);
    /* A single load spike does not drop to critical. */
    ps_power_update(&p, &cfg, 3000);
    CHECK(p.mode == PS_PWR_ECO);
    for (int i = 0; i < 60; i++) ps_power_update(&p, &cfg, 3300);
    CHECK(p.mode == PS_PWR_CRITICAL);
    for (int i = 0; i < 60; i++) ps_power_update(&p, &cfg, 3370); /* inside hysteresis */
    CHECK(p.mode == PS_PWR_CRITICAL);
}

int main(void)
{
    RUN(test_direction_phrases);
    RUN(test_queue_priority);
    RUN(test_nav_alert_levels);
    RUN(test_person_alert);
    RUN(test_power);
    TEST_MAIN_END();
}
