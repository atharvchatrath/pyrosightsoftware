#include "pyrosight/ps_nav.h"

#include <math.h>
#include <string.h>

#define PI_F 3.14159265358979f
#define DEG2RAD (PI_F / 180.0f)
#define RAD2DEG (180.0f / PI_F)

float ps_wrap_pi(float a)
{
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a <= -PI_F) a += 2.0f * PI_F;
    return a;
}

float ps_quat_to_yaw(float qw, float qx, float qy, float qz)
{
    /* Rotation about +Z (up), counter-clockwise positive: turning left
     * increases yaw. Matches the BNO085 game rotation vector frame. */
    return atan2f(2.0f * (qw * qz + qx * qy), 1.0f - 2.0f * (qy * qy + qz * qz));
}

static float dist2(ps_vec2_t a, ps_vec2_t b)
{
    float dx = a.x - b.x, dy = a.y - b.y;
    return sqrtf(dx * dx + dy * dy);
}

void ps_nav_init(ps_nav_t *nav)
{
    memset(nav, 0, sizeof(*nav));
    nav->state = PS_NAV_IDLE;
}

static void add_crumb(ps_nav_t *nav, const ps_config_t *cfg, uint32_t t_ms)
{
    if (nav->n_crumbs > 0 &&
        dist2(nav->pos, nav->crumbs[nav->n_crumbs - 1].p) < cfg->waypoint_reached_m)
        return; /* too close to the previous crumb to be useful */
    if (nav->n_crumbs >= 2) {
        /* Retracing: heading back toward the previous crumb is following the
         * trail out, not extending it. Appending here would put a crumb
         * between the wearer and the door and swing the arrow backwards. */
        ps_vec2_t last = nav->crumbs[nav->n_crumbs - 1].p, prev = nav->crumbs[nav->n_crumbs - 2].p;
        if (dist2(nav->pos, prev) < dist2(last, prev)) return;
    }
    if (nav->n_crumbs >= PS_NAV_MAX_CRUMBS) {
        /* Full: thin the trail by dropping every other crumb after the door. */
        uint16_t w = 1;
        for (uint16_t r = 2; r < nav->n_crumbs; r += 2) nav->crumbs[w++] = nav->crumbs[r];
        nav->n_crumbs = w;
    }
    nav->crumbs[nav->n_crumbs].p = nav->pos;
    nav->crumbs[nav->n_crumbs].t_ms = t_ms;
    nav->crumbs[nav->n_crumbs].heading_sigma = nav->heading_sigma_rad;
    nav->n_crumbs++;
}

/*
 * Loop removal: if the wearer is back within reach of an older crumb,
 * everything dropped after it is a detour and is pruned. This is what makes
 * the trail shrink as they walk out, and what lets a wander-around-a-room
 * collapse into the shortest known way back.
 */
static void prune_loops(ps_nav_t *nav, const ps_config_t *cfg)
{
    if (nav->n_crumbs < 2) return;
    for (uint16_t i = 0; i + 1 < nav->n_crumbs; i++) {
        if (dist2(nav->pos, nav->crumbs[i].p) < cfg->waypoint_reached_m) {
            nav->n_crumbs = (uint16_t)(i + 1);
            return;
        }
    }
}

static void update_return_target(ps_nav_t *nav, const ps_config_t *cfg)
{
    if (nav->n_crumbs == 0) { nav->return_target = 0; return; }
    uint16_t last = (uint16_t)(nav->n_crumbs - 1);
    if (last == 0) { nav->return_target = 0; return; }
    /* Point past the last crumb once the wearer is standing on it, or is
     * already on the way from it toward the one before. */
    const ps_vec2_t lp = nav->crumbs[last].p, pp = nav->crumbs[last - 1].p;
    bool on_last = dist2(nav->pos, lp) < cfg->waypoint_reached_m;
    bool heading_back = dist2(nav->pos, pp) < dist2(lp, pp);
    nav->return_target = (on_last || heading_back) ? (uint16_t)(last - 1) : last;
}

static float yaw_change_in_window(const ps_nav_t *nav, uint32_t t_ms, uint32_t window_ms)
{
    /* Signed yaw change from the oldest sample inside the window to now. */
    if (nav->yaw_hist_n < 2) return 0.0f;
    int newest = (nav->yaw_hist_head + PS_NAV_YAW_HISTORY - 1) % PS_NAV_YAW_HISTORY;
    float oldest_yaw = nav->yaw_hist[newest];
    for (int k = 1; k < nav->yaw_hist_n; k++) {
        int idx = (newest - k + PS_NAV_YAW_HISTORY) % PS_NAV_YAW_HISTORY;
        if (t_ms - nav->yaw_hist_t[idx] > window_ms) break;
        oldest_yaw = nav->yaw_hist[idx];
    }
    return ps_wrap_pi(nav->yaw_hist[newest] - oldest_yaw);
}

void ps_nav_mark_entry(ps_nav_t *nav, uint32_t t_ms)
{
    float yaw_raw = nav->yaw_raw;
    bool have_yaw = nav->have_yaw;
    ps_nav_init(nav);
    nav->yaw_raw = yaw_raw;
    nav->have_yaw = have_yaw;
    nav->yaw_offset = -yaw_raw; /* entry heading becomes 0 */
    nav->yaw = 0.0f;
    nav->state = PS_NAV_TRACKING;
    nav->t_entry_ms = nav->t_last_ms = nav->t_last_imu_ms = t_ms;
    nav->confidence = 1.0f;
    nav->crumbs[0].p = nav->pos;
    nav->crumbs[0].t_ms = t_ms;
    nav->n_crumbs = 1;
}

void ps_nav_on_yaw(ps_nav_t *nav, const ps_config_t *cfg, float yaw_rad, uint32_t t_ms)
{
    nav->yaw_raw = yaw_rad;
    nav->have_yaw = true;
    nav->t_last_imu_ms = t_ms;
    if (nav->state == PS_NAV_LOST) nav->state = PS_NAV_TRACKING;
    nav->yaw = ps_wrap_pi(yaw_rad + nav->yaw_offset);
    if (nav->state != PS_NAV_TRACKING) return;

    /* Keep ~1 sample per 100 ms in the turn-detection history. */
    int newest = (nav->yaw_hist_head + PS_NAV_YAW_HISTORY - 1) % PS_NAV_YAW_HISTORY;
    if (nav->yaw_hist_n == 0 || t_ms - nav->yaw_hist_t[newest] >= 100) {
        nav->yaw_hist[nav->yaw_hist_head] = nav->yaw;
        nav->yaw_hist_t[nav->yaw_hist_head] = t_ms;
        nav->yaw_hist_head = (uint8_t)((nav->yaw_hist_head + 1) % PS_NAV_YAW_HISTORY);
        if (nav->yaw_hist_n < PS_NAV_YAW_HISTORY) nav->yaw_hist_n++;
    }

    float dyaw = yaw_change_in_window(nav, t_ms, (uint32_t)cfg->turn_window_ms);
    if (fabsf(dyaw) * RAD2DEG >= cfg->turn_threshold_deg &&
        t_ms - nav->t_last_turn_ms > (uint32_t)cfg->turn_window_ms) {
        nav->turns++;
        nav->t_last_turn_ms = t_ms;
        nav->last_turn_deg = dyaw * RAD2DEG;
        /* Each turn adds a little heading uncertainty (scale-factor error). */
        nav->heading_sigma_rad += 0.01f * fabsf(dyaw);
        add_crumb(nav, cfg, t_ms);
        update_return_target(nav, cfg);
    }
}

void ps_nav_on_step(ps_nav_t *nav, const ps_config_t *cfg, uint32_t t_ms)
{
    if (nav->state != PS_NAV_TRACKING) return;
    const float L = cfg->step_length_m;

    /* Heuristic drift elimination: corridors and rooms are mostly aligned to
     * the doorway wall. When walking straight close to one of the four axes,
     * nudge the heading toward it to cancel slow gyro drift. */
    if (cfg->heading_snap) {
        float straight = fabsf(yaw_change_in_window(nav, t_ms, 1000)) * RAD2DEG;
        if (straight < 5.0f) {
            float axis = roundf(nav->yaw / (PI_F / 2)) * (PI_F / 2);
            float err = ps_wrap_pi(axis - nav->yaw);
            if (fabsf(err) * RAD2DEG < cfg->heading_snap_deg) {
                nav->yaw_offset += err * cfg->heading_snap_gain;
                nav->yaw = ps_wrap_pi(nav->yaw_raw + nav->yaw_offset);
            }
        }
    }

    nav->pos.x += L * cosf(nav->yaw);
    nav->pos.y += L * sinf(nav->yaw);
    nav->dist_walked_m += L;
    nav->steps++;
    nav->t_last_step_ms = t_ms;

    const float sl = cfg->step_length_sigma * L;
    nav->var_steps_m2 += sl * sl;

    float since = nav->n_crumbs ? dist2(nav->pos, nav->crumbs[nav->n_crumbs - 1].p) : 0.0f;
    prune_loops(nav, cfg);
    if (since >= cfg->breadcrumb_spacing_m) add_crumb(nav, cfg, t_ms);
    update_return_target(nav, cfg);
}

void ps_nav_on_linear_accel(ps_nav_t *nav, const ps_config_t *cfg, float mag, uint32_t t_ms)
{
    (void)cfg;
    nav->t_last_imu_ms = t_ms;
    nav->accel_lp += 0.1f * (mag - nav->accel_lp);
}

void ps_nav_tick(ps_nav_t *nav, const ps_config_t *cfg, uint32_t t_ms)
{
    if (nav->state == PS_NAV_IDLE) { nav->t_last_ms = t_ms; return; }
    float dt = (float)(t_ms - nav->t_last_ms) * 0.001f;
    nav->t_last_ms = t_ms;
    if (dt <= 0.0f) return;

    if (t_ms - nav->t_last_imu_ms > cfg->imu_timeout_ms) nav->state = PS_NAV_LOST;

    /* Heading drift accumulates with time. */
    nav->heading_sigma_rad += cfg->gyro_drift_deg_per_min * DEG2RAD * dt / 60.0f;

    if (nav->state == PS_NAV_LOST) {
        /* No IMU: they may have moved at a brisk walk in any direction. Kept
         * deliberately pessimistic: in simulation a 4 s outage that went
         * unmeasured made the wearer miss a doorway while the arrow was still
         * shown, so an outage of a few seconds should end in "follow hose". */
        float sd = sqrtf(nav->var_events_m2) + 1.5f * dt;
        nav->var_events_m2 = sd * sd;
    } else if (nav->accel_lp > 1.2f && t_ms - nav->t_last_step_ms > 1500) {
        /* Moving without counted steps: crawling (the standard posture in
         * heavy smoke), dragging a casualty, climbing. Advance at a nominal
         * crawl speed along the heading, and grow the uncertainty with the
         * full speed error since we cannot measure it. */
        nav->untracked_s += dt;
        float v = cfg->crawl_speed_mps * dt;
        nav->pos.x += v * cosf(nav->yaw);
        nav->pos.y += v * sinf(nav->yaw);
        nav->dist_walked_m += v;
        float sd = sqrtf(nav->var_events_m2) + cfg->missed_motion_sigma_m_per_s * dt;
        nav->var_events_m2 = sd * sd;
        float since = nav->n_crumbs ? dist2(nav->pos, nav->crumbs[nav->n_crumbs - 1].p) : 0.0f;
        prune_loops(nav, cfg);
        if (since >= cfg->breadcrumb_spacing_m) add_crumb(nav, cfg, t_ms);
        update_return_target(nav, cfg);
    }

    /*
     * Uncertainty relative to the way out, not in absolute terms. The trail
     * and the position estimate share the same stride and heading errors, so
     * retracing cancels most of them. What remains:
     *   - stride bias over the route still ahead;
     *   - heading drift since each remaining crumb was dropped, times the
     *     length of route that depends on it (cross-track error);
     *   - per-step noise, in proportion to the route ahead;
     *   - displacement that was never measured (crawling, IMU outage).
     */
    float remaining = 0.0f, cross = 0.0f;
    if (nav->n_crumbs) {
        const ps_crumb_t *tc = &nav->crumbs[nav->return_target];
        float d = dist2(nav->pos, tc->p);
        remaining = d;
        cross = d * fmaxf(0.0f, nav->heading_sigma_rad - tc->heading_sigma);
        for (int i = nav->return_target; i > 0; i--) {
            float seg = dist2(nav->crumbs[i].p, nav->crumbs[i - 1].p);
            remaining += seg;
            cross += seg * fmaxf(0.0f, nav->heading_sigma_rad - nav->crumbs[i].heading_sigma);
        }
    }
    nav->cross_sigma_m = cross;
    const float frac = nav->dist_walked_m > 0.0f ? fminf(1.0f, remaining / nav->dist_walked_m) : 0.0f;
    const float bias = cfg->step_length_bias * remaining;
    float sigma = sqrtf(nav->var_steps_m2 * frac + nav->var_events_m2 + bias * bias + cross * cross);
    nav->pos_sigma_m = sigma;
    float r = sigma / cfg->nav_conf_scale_m;
    nav->confidence = 1.0f / (1.0f + r * r);
}

void ps_nav_guidance(const ps_nav_t *nav, ps_nav_guidance_t *g)
{
    memset(g, 0, sizeof(*g));
    if (nav->state == PS_NAV_IDLE || nav->n_crumbs == 0) return;
    g->valid = true;

    const ps_vec2_t door = nav->crumbs[0].p;
    g->home_dist_m = dist2(nav->pos, door);
    float home_abs = atan2f(door.y - nav->pos.y, door.x - nav->pos.x);
    g->home_bearing_rel_deg = ps_wrap_pi(home_abs - nav->yaw) * RAD2DEG;

    const ps_vec2_t tgt = nav->crumbs[nav->return_target].p;
    float tgt_abs = atan2f(tgt.y - nav->pos.y, tgt.x - nav->pos.x);
    g->route_bearing_rel_deg = ps_wrap_pi(tgt_abs - nav->yaw) * RAD2DEG;
    float d = dist2(nav->pos, tgt);
    for (int i = nav->return_target; i > 0; i--) d += dist2(nav->crumbs[i].p, nav->crumbs[i - 1].p);
    g->route_dist_m = d;

    g->pos_sigma_m = nav->pos_sigma_m;
    g->confidence = nav->confidence;
}
