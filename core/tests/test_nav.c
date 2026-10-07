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


/* ---------------------------------------------------------------- crawling */

/* The hand-knee gait a crawling wearer puts into the accelerometer: an impact
 * every stride, decaying over ~120 ms, on a low baseline. Same shape the
 * simulator generates (sim/sim_world.c). */
static void feed_crawl(ps_nav_t *n, const ps_config_t *c, uint32_t *clock,
                       float seconds, float speed_mps, float stride_m)
{
    float phase = 0.0f;
    int ticks = (int)(seconds * 100.0f);
    for (int i = 0; i < ticks; i++) {
        *clock += 10;
        phase += (speed_mps / stride_m) * 0.01f;
        float impact = 0.0f;
        if (phase >= 1.0f) { phase -= 1.0f; impact = 1.0f; }
        float tail = expf(-phase * stride_m / speed_mps / 0.12f);
        float a = 1.1f + 2.6f * (impact > 0.0f ? 1.0f : tail) + 0.05f * ps_test_gauss();
        ps_nav_on_yaw(n, c, 0.0f, *clock);
        ps_nav_on_linear_accel(n, c, fabsf(a), *clock);
        ps_nav_tick(n, c, *clock);
    }
}

static void test_crawl_strides_measure_distance(void)
{
    /* The walking step detector is blind to crawling, so before this the
     * device could only assume a speed and multiply by elapsed time. Counting
     * hand-knee cycles measures the distance instead, which is what keeps the
     * arrow usable on a route crawled in heavy smoke. */
    ps_config_default(&cfg);
    ps_nav_init(&nav);
    now = 1000;
    ps_nav_on_yaw(&nav, &cfg, 0.0f, now);
    ps_nav_mark_entry(&nav, now);

    const float speed = 0.35f, stride = 0.45f, secs = 40.0f;
    feed_crawl(&nav, &cfg, &now, secs, speed, stride);

    float truth = speed * secs;
    printf("  crawled %.1f m, device measured %.2f m over %u strides\n",
           truth, nav.pos.x, nav.crawl_strides);
    CHECK(nav.crawl_strides > 25);
    /* Within 15%: the assumed-speed model it replaces was 14% short on its
     * own nominal, and got worse as soon as the wearer changed pace. */
    CHECK(fabsf(nav.pos.x - truth) < 0.15f * truth);
    CHECK(nav.confidence > 0.5f);
}

static void test_crawl_speed_change_is_tracked(void)
{
    /* Counting strides, not clock ticks: crawl slowly and the device must
     * report less distance, where an assumed speed would report the same. */
    ps_config_default(&cfg);
    ps_nav_t fast_nav, slow_nav;
    uint32_t t1 = 1000, t2 = 1000;

    ps_nav_init(&fast_nav);
    ps_nav_on_yaw(&fast_nav, &cfg, 0.0f, t1); ps_nav_mark_entry(&fast_nav, t1);
    feed_crawl(&fast_nav, &cfg, &t1, 30.0f, 0.45f, 0.45f);

    ps_nav_init(&slow_nav);
    ps_nav_on_yaw(&slow_nav, &cfg, 0.0f, t2); ps_nav_mark_entry(&slow_nav, t2);
    feed_crawl(&slow_nav, &cfg, &t2, 30.0f, 0.22f, 0.45f);

    printf("  same 30 s: fast crawl %.2f m, slow crawl %.2f m\n", fast_nav.pos.x, slow_nav.pos.x);
    CHECK(fast_nav.pos.x > slow_nav.pos.x * 1.5f);
}

static void test_isolated_impacts_are_not_crawling(void)
{
    /* A dropped tool, a bumped doorframe, or simply starting to walk after
     * standing still all spike the accelerometer once. Crediting distance for
     * a single spike invented 8.9 m on a route where nobody crawled, so a
     * cadence has to be established first. */
    ps_config_default(&cfg);
    ps_nav_init(&nav);
    now = 1000;
    ps_nav_on_yaw(&nav, &cfg, 0.0f, now);
    ps_nav_mark_entry(&nav, now);

    for (int bump = 0; bump < 8; bump++) {
        for (int i = 0; i < 400; i++) {          /* 4 s of near-stillness */
            now += 10;
            ps_nav_on_yaw(&nav, &cfg, 0.0f, now);
            ps_nav_on_linear_accel(&nav, &cfg, 0.2f + 0.02f * ps_test_gauss(), now);
            ps_nav_tick(&nav, &cfg, now);
        }
        for (int i = 0; i < 8; i++) {            /* one isolated 80 ms knock */
            now += 10;
            ps_nav_on_yaw(&nav, &cfg, 0.0f, now);
            ps_nav_on_linear_accel(&nav, &cfg, 4.0f, now);
            ps_nav_tick(&nav, &cfg, now);
        }
    }
    printf("  8 isolated knocks produced %.2f m of phantom distance\n", nav.crawl_dist_m);
    CHECK(nav.crawl_strides == 0);
    CHECK(nav.crawl_dist_m < 0.5f);
}

/* ------------------------------------------------------------- IMU outages */

static void test_coasts_through_imu_dropout(void)
{
    /* The sensor dies for three seconds while the wearer is walking. Freezing
     * the position assumes they stopped dead the instant it failed, which is
     * the one thing they certainly did not do. */
    ps_config_default(&cfg);
    ps_nav_init(&nav);
    now = 1000;
    ps_nav_on_yaw(&nav, &cfg, 0.0f, now);
    ps_nav_mark_entry(&nav, now);

    /* Walk 12 steps at 1.6 steps/s to establish a cadence. */
    for (int i = 0; i < 12; i++) {
        for (int k = 0; k < 62; k++) {
            now += 10;
            ps_nav_on_yaw(&nav, &cfg, 0.0f, now);
            ps_nav_on_linear_accel(&nav, &cfg, 2.2f, now);
            ps_nav_tick(&nav, &cfg, now);
        }
        ps_nav_on_step(&nav, &cfg, now);
    }
    float before = nav.pos.x;

    /* Three seconds with no IMU at all: only the clock still runs. */
    for (int k = 0; k < 300; k++) { now += 10; ps_nav_tick(&nav, &cfg, now); }

    float coasted = nav.pos.x - before;
    float truth = 0.62f * 1.6f * 3.0f;   /* stride * cadence * seconds */
    printf("  3 s outage: coasted %.2f m, wearer really covered ~%.2f m\n", coasted, truth);
    CHECK(nav.state == PS_NAV_LOST);
    CHECK(nav.gaps == 1);
    CHECK(coasted > 0.5f * truth);
    CHECK(coasted < 1.5f * truth);
    /* Honest about it: coasting is a prediction, so uncertainty still grows. */
    CHECK(nav.pos_sigma_m > 1.0f);
}

static void test_coasting_stops_when_the_wearer_had_stopped(void)
{
    /* If they were standing still when the sensor died, coasting must not
     * walk them across the room. */
    ps_config_default(&cfg);
    ps_nav_init(&nav);
    now = 1000;
    ps_nav_on_yaw(&nav, &cfg, 0.0f, now);
    ps_nav_mark_entry(&nav, now);
    for (int i = 0; i < 6; i++) {
        for (int k = 0; k < 62; k++) { now += 10; ps_nav_on_yaw(&nav, &cfg, 0.0f, now); ps_nav_tick(&nav, &cfg, now); }
        ps_nav_on_step(&nav, &cfg, now);
    }
    /* Five seconds standing still, then the IMU dies. */
    for (int k = 0; k < 500; k++) {
        now += 10;
        ps_nav_on_yaw(&nav, &cfg, 0.0f, now);
        ps_nav_on_linear_accel(&nav, &cfg, 0.2f, now);
        ps_nav_tick(&nav, &cfg, now);
    }
    float before = nav.pos.x;
    for (int k = 0; k < 300; k++) { now += 10; ps_nav_tick(&nav, &cfg, now); }
    printf("  stationary outage coasted %.2f m\n", nav.pos.x - before);
    CHECK(fabsf(nav.pos.x - before) < 0.3f);
}

static void test_snap_restores_heading_confidence(void)
{
    /* The snap is an observation, not just a correction: walking straight
     * along a building axis bounds the heading error, so the uncertainty has
     * to come back down. Left growing, it reached 35 degrees on the long
     * route while the real error stayed near 2, and the device disowned a
     * position estimate that was good to a metre. */
    ps_config_default(&cfg);
    ps_nav_init(&nav);
    now = 1000;
    ps_nav_on_yaw(&nav, &cfg, 0.0f, now);
    ps_nav_mark_entry(&nav, now);

    /* Six legs with 90-degree turns between them, gyro drifting throughout:
     * turns and time are both sources of heading uncertainty, and a long
     * search is made of them. */
    float yaw = 0.0f, drift = 0.0f;
    for (int leg = 0; leg < 16; leg++) {
        for (int i = 0; i < 25; i++) {
            for (int k = 0; k < 62; k++) {
                now += 10; drift += 0.00002f;
                ps_nav_on_yaw(&nav, &cfg, yaw + drift, now);
                ps_nav_tick(&nav, &cfg, now);
            }
            ps_nav_on_step(&nav, &cfg, now);
        }
        for (int k = 0; k < 100; k++) {      /* turn 90 degrees over 1 s */
            now += 10; yaw += (PI / 2) / 100.0f;
            ps_nav_on_yaw(&nav, &cfg, yaw + drift, now);
            ps_nav_tick(&nav, &cfg, now);
        }
    }
    float snapped = nav.heading_sigma_rad;

    cfg.heading_snap = false;
    ps_nav_t alt; ps_nav_init(&alt);
    now = 1000; yaw = 0.0f; drift = 0.0f;
    ps_nav_on_yaw(&alt, &cfg, 0.0f, now);
    ps_nav_mark_entry(&alt, now);
    for (int leg = 0; leg < 16; leg++) {
        for (int i = 0; i < 25; i++) {
            for (int k = 0; k < 62; k++) {
                now += 10; drift += 0.00002f;
                ps_nav_on_yaw(&alt, &cfg, yaw + drift, now);
                ps_nav_tick(&alt, &cfg, now);
            }
            ps_nav_on_step(&alt, &cfg, now);
        }
        for (int k = 0; k < 100; k++) {
            now += 10; yaw += (PI / 2) / 100.0f;
            ps_nav_on_yaw(&alt, &cfg, yaw + drift, now);
            ps_nav_tick(&alt, &cfg, now);
        }
    }
    printf("  heading sigma after a 16-leg search: %.1f deg with snap, %.1f without\n",
           snapped * 180.0f / PI, alt.heading_sigma_rad * 180.0f / PI);
    CHECK(snapped < alt.heading_sigma_rad * 0.7f);
    CHECK(nav.confidence > alt.confidence);
}

int main(void)
{
    RUN(test_quat_yaw);
    RUN(test_l_shaped_walk_and_way_back);
    RUN(test_confidence_decays_and_imu_loss);
    RUN(test_crawling_lowers_confidence);
    RUN(test_heading_snap_cancels_drift);
    RUN(test_snap_restores_heading_confidence);
    RUN(test_crawl_strides_measure_distance);
    RUN(test_crawl_speed_change_is_tracked);
    RUN(test_isolated_impacts_are_not_crawling);
    RUN(test_coasts_through_imu_dropout);
    RUN(test_coasting_stops_when_the_wearer_had_stopped);
    TEST_MAIN_END();
}
