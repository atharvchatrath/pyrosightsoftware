/*
 * PyroSight Camera demo navigation: C API shim over the device's own
 * navigation code (core/src/ps_nav.c, ps_config.c, ps_alerts.c).
 *
 * The same file is compiled twice:
 *   - to WebAssembly (zig cc, wasm32-wasi, MVP) and then to plain JavaScript
 *     with binaryen's wasm2js, for the browser (build_nav.py);
 *   - natively with gcc, for the parity test (tests/parity_native.c).
 *
 * It mirrors what ps_system.c does around navigation (mark entry resets the
 * alert state and says "Entry point marked", tick = nav tick + guidance +
 * alert levels with the device's hysteresis), without the thermal pipeline.
 * All state is static; every export is a plain C function.
 */
#include <math.h>
#include <string.h>

#include "pyrosight/ps_alerts.h"
#include "pyrosight/ps_config.h"
#include "pyrosight/ps_nav.h"

#ifdef __wasm__
#define API __attribute__((visibility("default")))
#else
#define API
#endif

static ps_config_t cfg;
static ps_nav_t nav;
static ps_alerts_t alerts;
static float st[32];
/* Motion-sensor health, as in ps_system_tick(): a stream that stops is announced once. */
static bool imu_seen, imu_ok;
static uint32_t t_last_imu;
static void imu_alive(uint32_t t_ms) { t_last_imu = t_ms; imu_seen = true; imu_ok = true; }
static float crumbs[PS_NAV_MAX_CRUMBS * 3];
static int32_t alert_out[8];

API void api_init(void)
{
    ps_config_default(&cfg);
    ps_nav_init(&nav);
    ps_alerts_init(&alerts);
    imu_seen = imu_ok = false;
    t_last_imu = 0;
}

/* Configuration overrides (key ids documented in nav_core.js CFG). */
API void api_set_cfg(int key, float v)
{
    switch (key) {
    case 0: cfg.step_length_m = v; break;
    case 1: cfg.heading_snap = v != 0.0f; break;
    case 2: cfg.imu_timeout_ms = (uint32_t)v; break;
    case 3: cfg.gyro_drift_deg_per_min = v; break;
    case 4: cfg.nav_conf_scale_m = v; break;
    case 5: cfg.waypoint_reached_m = v; break;
    case 6: cfg.breadcrumb_spacing_m = v; break;
    default: break;
    }
}

API float api_get_cfg(int key)
{
    switch (key) {
    case 0: return cfg.step_length_m;
    case 1: return cfg.heading_snap ? 1.0f : 0.0f;
    case 2: return (float)cfg.imu_timeout_ms;
    case 3: return cfg.gyro_drift_deg_per_min;
    case 4: return cfg.nav_conf_scale_m;
    case 5: return cfg.waypoint_reached_m;
    case 6: return cfg.breadcrumb_spacing_m;
    case 7: return cfg.nav_conf_warn;
    case 8: return cfg.nav_conf_unreliable;
    case 9: return cfg.camera_height_m;
    default: return 0.0f;
    }
}

API void api_mark_entry(uint32_t t_ms)
{
    ps_nav_mark_entry(&nav, t_ms);
    ps_alerts_init(&alerts);
    ps_alerts_push(&alerts, PS_PRIO_INFO, t_ms, PS_PHRASE_ENTRY_MARKED, PS_PHRASE_NONE, PS_PHRASE_NONE);
}

/* Back to "entry not marked" (keeps the last yaw sample). */
API void api_reset(void)
{
    float yr = nav.yaw_raw;
    bool hy = nav.have_yaw;
    ps_nav_init(&nav);
    nav.yaw_raw = yr;
    nav.have_yaw = hy;
    ps_alerts_init(&alerts);
}

API void api_on_yaw(float yaw_rad, uint32_t t_ms) { imu_alive(t_ms); ps_nav_on_yaw(&nav, &cfg, yaw_rad, t_ms); }
API void api_on_step(uint32_t t_ms) { ps_nav_on_step(&nav, &cfg, t_ms); }
API void api_on_accel(float mag, uint32_t t_ms) { imu_alive(t_ms); ps_nav_on_linear_accel(&nav, &cfg, mag, t_ms); }

/* Same order as ps_system_tick(): nav tick, guidance, navigation alerts. */
API void api_tick(uint32_t t_ms)
{
    /* ps_system_tick()'s health supervisor, motion-sensor part: "Motion sensor lost." once, while
     * navigating (the alert queue is reset when the entry is marked). */
    if (imu_ok && imu_seen && nav.state != PS_NAV_IDLE && t_ms - t_last_imu > cfg.imu_timeout_ms) {
        imu_ok = false;
        ps_alerts_push(&alerts, PS_PRIO_WARNING, t_ms, PS_PHRASE_MOTION_LOST, PS_PHRASE_NONE, PS_PHRASE_NONE);
    }
    ps_nav_tick(&nav, &cfg, t_ms);
    ps_nav_guidance_t g;
    ps_nav_guidance(&nav, &g);
    ps_alerts_update_nav(&alerts, &cfg, &g, t_ms);
}

/* The wearer asked "where is out?" (PS_BTN_WHERE_OUT). */
API void api_where_out(uint32_t t_ms)
{
    ps_nav_guidance_t g;
    ps_nav_guidance(&nav, &g);
    ps_alerts_request_direction(&alerts, &g, t_ms);
}

/* State + guidance snapshot. Layout = nav_core.js S. */
API float *api_state(void)
{
    ps_nav_guidance_t g;
    ps_nav_guidance(&nav, &g);
    float *s = st;
    s[0] = (float)nav.state;
    s[1] = g.valid ? 1.0f : 0.0f;
    s[2] = g.route_bearing_rel_deg;
    s[3] = g.home_bearing_rel_deg;
    s[4] = g.route_dist_m;
    s[5] = g.home_dist_m;
    s[6] = g.pos_sigma_m;
    s[7] = g.confidence;
    s[8] = g.exit_is_next ? 1.0f : 0.0f;
    s[9] = (float)alerts.level;
    s[10] = nav.pos.x;
    s[11] = nav.pos.y;
    s[12] = nav.yaw;
    s[13] = nav.dist_walked_m;
    s[14] = (float)nav.steps;
    s[15] = (float)nav.n_crumbs;
    s[16] = (float)nav.return_target;
    s[17] = nav.heading_sigma_rad;
    s[18] = (float)nav.turns;
    s[19] = (float)ps_direction_phrase(g.route_bearing_rel_deg);
    s[20] = (float)ps_direction_phrase(g.home_bearing_rel_deg);
    s[21] = nav.untracked_s;
    s[22] = nav.accel_lp;
    s[23] = nav.cross_sigma_m;
    s[24] = nav.have_yaw ? 1.0f : 0.0f;
    s[25] = nav.yaw_raw;
    return st;
}

/* Breadcrumbs, entry frame (+x = heading at entry, +y = left): x, y, heading_sigma. */
API float *api_crumbs(void)
{
    for (int i = 0; i < nav.n_crumbs; i++) {
        crumbs[3 * i] = nav.crumbs[i].p.x;
        crumbs[3 * i + 1] = nav.crumbs[i].p.y;
        crumbs[3 * i + 2] = nav.crumbs[i].heading_sigma;
    }
    return crumbs;
}
API int api_n_crumbs(void) { return nav.n_crumbs; }

/* Pop the next spoken alert: returns the number of phrase parts (0 = none);
 * api_alert_buf() then holds parts[0..3], prio, t_ms. */
API int api_alert_pop(void)
{
    ps_alert_t a;
    if (!ps_alerts_pop(&alerts, &a)) return 0;
    for (int i = 0; i < PS_ALERT_MAX_PARTS; i++) alert_out[i] = i < a.n_parts ? (int32_t)a.parts[i] : 0;
    alert_out[4] = (int32_t)a.prio;
    alert_out[5] = (int32_t)a.t_ms;
    return a.n_parts;
}
API int32_t *api_alert_buf(void) { return alert_out; }

API int api_direction_phrase(float rel_bearing_deg) { return (int)ps_direction_phrase(rel_bearing_deg); }
API const char *api_phrase_text(int id) { return id >= 0 && id < PS_PHRASE_COUNT ? ps_phrase_text[id] : ""; }
API int api_phrase_count(void) { return PS_PHRASE_COUNT; }
API float api_quat_to_yaw(float qw, float qx, float qy, float qz) { return ps_quat_to_yaw(qw, qx, qy, qz); }
API float api_wrap_pi(float a) { return ps_wrap_pi(a); }
