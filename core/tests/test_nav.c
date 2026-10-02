#include "ps_test.h"
#include "pyrosight/ps_nav.h"

#define PI 3.14159265f

static ps_config_t cfg;
static ps_nav_t nav;
static uint32_t now;

/* Walk n steps at a fixed heading, feeding yaw at 100 Hz like the BNO085. */
static void walk(int n, float yaw)
{
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < 50; k++) {
            now += 10;
            ps_nav_on_yaw(&nav, &cfg, yaw, now);
            ps_nav_on_linear_accel(&nav, &cfg, 2.0f, now);
            ps_nav_tick(&nav, &cfg, now);
        }
        ps_nav_on_step(&nav, &cfg, now);
    }
}

/* Turn in place over one second. */
static void turn(float from, float to)
{
    for (int k = 1; k <= 100; k++) {
        now += 10;
        ps_nav_on_yaw(&nav, &cfg, from + (to - from) * k / 100.0f, now);
        ps_nav_tick(&nav, &cfg, now);
    }
}

static void setup(void)
{
    ps_config_default(&cfg);
    cfg.heading_snap = false;
    ps_nav_init(&nav);
    now = 1000;
    ps_nav_on_yaw(&nav, &cfg, 0.7f, now); /* arbitrary absolute heading at the door */
    ps_nav_mark_entry(&nav, now);
}

static void test_quat_yaw(void)
{
    float a = 0.9f; /* rotation about Z */
    CHECK_NEAR(ps_quat_to_yaw(cosf(a / 2), 0, 0, sinf(a / 2)), 0.9, 1e-5);
    CHECK_NEAR(ps_wrap_pi(3 * PI), PI, 1e-5);
    CHECK_NEAR(ps_wrap_pi(-PI / 2 - 2 * PI), -PI / 2, 1e-5);
}

static void test_l_shaped_walk_and_way_back(void)
{
    setup();
    const float L = cfg.step_length_m;
    walk(10, 0.7f);
    turn(0.7f, 0.7f + PI / 2); /* turn left */
    walk(10, 0.7f + PI / 2);
    CHECK_NEAR(nav.pos.x, 10 * L, 0.05);
    CHECK_NEAR(nav.pos.y, 10 * L, 0.05);
    CHECK(nav.turns == 1);
    CHECK(nav.state == PS_NAV_TRACKING);

    ps_nav_guidance_t g;
    ps_nav_guidance(&nav, &g);
    /* Facing +y after the left turn; the route goes back to the corner, i.e. behind. */
    CHECK(fabsf(g.route_bearing_rel_deg) > 170.0f);
    CHECK_NEAR(g.route_dist_m, 20 * L, 0.2);
    CHECK_NEAR(g.home_dist_m, sqrtf(2.0f) * 10 * L, 0.1);
    /* Facing +y, the door (origin) is behind and to the left: about +135 deg. */
    CHECK_NEAR(g.home_bearing_rel_deg, 135.0, 2.0);
    printf("  after 20 steps: sigma %.2f m, confidence %.2f, crumbs %d\n", g.pos_sigma_m, g.confidence, nav.n_crumbs);
    CHECK(g.confidence > 0.6f && g.confidence < 1.0f);

    /* Turn around and walk back to the corner: the arrow should then swing to
     * point down the first corridor, and the trail should shrink. */
    turn(0.7f + PI / 2, 0.7f + 3 * PI / 2);
    walk(10, 0.7f + 3 * PI / 2);
    ps_nav_guidance(&nav, &g);
    CHECK_NEAR(g.route_dist_m, 10 * L, 1.0);
    turn(0.7f - PI / 2, 0.7f - PI);
    ps_nav_guidance(&nav, &g);
    CHECK(fabsf(g.route_bearing_rel_deg) < 15.0f); /* door is straight ahead now */
    walk(10, 0.7f - PI);
    ps_nav_guidance(&nav, &g);
    CHECK(g.home_dist_m < 0.5f);
    CHECK(g.route_dist_m < 1.0f);
    CHECK(nav.n_crumbs <= 2);
}

static void test_confidence_decays_and_imu_loss(void)
{
    setup();
    walk(5, 0.7f);
    ps_nav_guidance_t g0, g1;
    ps_nav_guidance(&nav, &g0);
    walk(80, 0.7f);
    ps_nav_guidance(&nav, &g1);
    CHECK(g1.confidence < g0.confidence);

    /* IMU goes silent: state LOST and confidence collapses. */
    float c = nav.confidence;
    for (int k = 0; k < 600; k++) { now += 50; ps_nav_tick(&nav, &cfg, now); }
    CHECK(nav.state == PS_NAV_LOST);
    CHECK(nav.confidence < c * 0.5f);
    ps_nav_on_yaw(&nav, &cfg, 0.7f, now);
    CHECK(nav.state == PS_NAV_TRACKING);
}

static void test_crawling_lowers_confidence(void)
{
    setup();
    walk(3, 0.7f);
    float c0 = nav.confidence;
    /* 20 s of vigorous motion with no counted steps. */
    for (int k = 0; k < 2000; k++) {
        now += 10;
        ps_nav_on_yaw(&nav, &cfg, 0.7f, now);
        ps_nav_on_linear_accel(&nav, &cfg, 2.5f, now);
        ps_nav_tick(&nav, &cfg, now);
    }
    CHECK(nav.untracked_s > 15.0f);
    CHECK(nav.confidence < c0 - 0.1f);
}

static void test_heading_snap_cancels_drift(void)
{
    ps_config_default(&cfg);
    ps_nav_init(&nav);
    now = 1000;
    ps_nav_on_yaw(&nav, &cfg, 0.0f, now);
    ps_nav_mark_entry(&nav, now);
    /* Gyro drifts 6 degrees while walking a straight corridor. */
    for (int i = 0; i < 40; i++) {
        float drift = (6.0f * PI / 180.0f) * i / 40.0f;
        for (int k = 0; k < 50; k++) { now += 10; ps_nav_on_yaw(&nav, &cfg, drift, now); ps_nav_tick(&nav, &cfg, now); }
        ps_nav_on_step(&nav, &cfg, now);
    }
    float with_snap = fabsf(nav.pos.y);
    cfg.heading_snap = false;
    ps_nav_init(&nav);
    now = 1000;
    ps_nav_on_yaw(&nav, &cfg, 0.0f, now);
    ps_nav_mark_entry(&nav, now);
    for (int i = 0; i < 40; i++) {
        float drift = (6.0f * PI / 180.0f) * i / 40.0f;
        for (int k = 0; k < 50; k++) { now += 10; ps_nav_on_yaw(&nav, &cfg, drift, now); ps_nav_tick(&nav, &cfg, now); }
        ps_nav_on_step(&nav, &cfg, now);
    }
    printf("  cross-track error after 40 steps: %.2f m with snap, %.2f m without\n", with_snap, fabsf(nav.pos.y));
    CHECK(with_snap < fabsf(nav.pos.y) * 0.6f);
}

int main(void)
{
    RUN(test_quat_yaw);
    RUN(test_l_shaped_walk_and_way_back);
    RUN(test_confidence_decays_and_imu_loss);
    RUN(test_crawling_lowers_confidence);
    RUN(test_heading_snap_cancels_drift);
    TEST_MAIN_END();
}
