#include "pyrosight/ps_nav.h"

#include <math.h>
#include <string.h>

#define PI_F 3.14159265358979f
#define DEG2RAD (PI_F / 180.0f)
#define RAD2DEG (180.0f / PI_F)

/* A snap observation bounds heading error to a fraction of the snap window:
 * the error must have been inside the window for the snap to fire at all, and
 * is on average well inside it. */
#define PS_SNAP_SIGMA_FRAC 0.5f

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
                /*
                 * The snap is not only a correction, it is an OBSERVATION.
                 * Walking straight along a corridor that lines up with a
                 * building axis bounds the heading error to the snap window:
                 * had the error been larger, the heading would have fallen
                 * outside the window and no snap would have happened.
                 *
                 * Without this the uncertainty grew forever while the actual
                 * heading error stayed near 2 degrees — on the long route
                 * heading sigma reached 35 degrees, the cross-track term
                 * dominated the budget, and the device disowned a position
                 * estimate that was good to a metre. Confidence has to fall
                 * when the heading is unobserved and recover when it is
                 * observed, or it is not a measure of anything.
                 */
                const float bound = cfg->heading_snap_deg * DEG2RAD * PS_SNAP_SIGMA_FRAC;
                if (nav->heading_sigma_rad > bound) {
                    nav->heading_sigma_rad = bound +
                        (nav->heading_sigma_rad - bound) * (1.0f - cfg->heading_snap_gain);
                }
            }
        }
    }

    nav->pos.x += L * cosf(nav->yaw);
    nav->pos.y += L * sinf(nav->yaw);
    nav->dist_walked_m += L;
    nav->steps++;
    /* Speed over the ground from the cadence, so an IMU outage can be coasted
     * through rather than treated as the wearer standing still. */
    if (nav->t_last_step_ms && t_ms > nav->t_last_step_ms) {
        float gap_s = (float)(t_ms - nav->t_last_step_ms) * 0.001f;
        if (gap_s > 0.15f && gap_s < 3.0f) {
            float v = L / gap_s;
            nav->speed_mps = nav->speed_mps > 0.0f ? nav->speed_mps + 0.3f * (v - nav->speed_mps) : v;
        } else if (gap_s >= 3.0f) {
            nav->speed_mps = 0.0f;   /* they had stopped */
        }
    }
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
    nav->t_last_imu_ms = t_ms;
    nav->accel_lp += 0.1f * (mag - nav->accel_lp);

    /*
     * Count crawl strides.
     *
     * Crawling is the standard posture in heavy smoke and the walking step
     * detector is blind to it, so the device used to fall back on "assume
     * 0.3 m/s and hope". That turns every variation in how fast somebody
     * crawls into position error, and it has no way of noticing when they
     * stop. Each hand-knee cycle is an impact the accelerometer sees plainly;
     * counting those measures the distance instead of assuming it, and the
     * stride length is a per-wearer calibration exactly like walking stride.
     *
     * Two time constants: a fast average to carry the impact, a slow baseline
     * it has to stand above. The armed flag enforces one count per cycle.
     */
    nav->accel_fast += 0.35f * (mag - nav->accel_fast);
    nav->accel_base += 0.01f * (mag - nav->accel_base);

    /*
     * Re-arm on the quiet half of the cycle, whatever the regime. Gating the
     * arming on "currently crawling" was circular: between impacts the
     * smoothed magnitude dips below the motion threshold, so the detector was
     * disarmed in exactly the window where it had to re-arm, and counted
     * nothing at all.
     */
    /*
     * A crawl impact must clear its own baseline both by an absolute margin
     * and by a FACTOR. The absolute margin alone let steady walking qualify:
     * on the long route, strides missed by the walking detector left gaps
     * that looked like crawling, and noise excursions were counted as hand
     * placements — 12 phantom strides, 8.9 m of invented distance on a route
     * where nobody ever went to their knees. Walking acceleration is steady,
     * so it fails the ratio test however long the gap between counted steps.
     */
    float thresh = nav->accel_base + cfg->crawl_peak_margin;
    float ratio_thresh = nav->accel_base * cfg->crawl_peak_ratio;
    if (ratio_thresh > thresh) thresh = ratio_thresh;
    if (nav->accel_fast < thresh * 0.85f) nav->crawl_armed = true;

    if (nav->state != PS_NAV_TRACKING) return;
    /* The regime test uses the slow baseline, not the fast average: the gait
     * makes the fast one dip every cycle, which is motion, not a pause. */
    bool crawling = nav->accel_base > cfg->crawl_accel_min &&
                    t_ms - nav->t_last_step_ms > cfg->crawl_after_step_ms;
    if (!crawling) return;
    if (!nav->crawl_armed || nav->accel_fast < thresh) return;
    if (t_ms - nav->t_last_crawl_stride_ms < cfg->crawl_refractory_ms) return;

    /*
     * One impact is not a crawl. Dropping a tool, bumping a doorframe, or
     * simply starting to walk after standing still all produce a single
     * spike above a baseline that has not caught up yet — that last one was
     * manufacturing 8.9 m of distance on a route where nobody crawled.
     *
     * Distance is credited only once a CADENCE is established: consecutive
     * impacts spaced like a hand-knee cycle. The first impact of a genuine
     * crawl is therefore not counted, which costs half a stride of lag and
     * is the right trade against inventing motion that never happened.
     */
    uint32_t gap_ms = t_ms - nav->t_last_crawl_stride_ms;
    bool in_cadence = nav->t_last_crawl_stride_ms != 0 && gap_ms <= cfg->crawl_max_interval_ms;
    nav->crawl_armed = false;
    nav->t_last_crawl_stride_ms = t_ms;
    if (!in_cadence) { nav->crawl_lock = 1; return; }
    if (nav->crawl_lock < 3) nav->crawl_lock++;
    if (nav->crawl_lock < 2) return;
    nav->crawl_strides++;

    const float L = cfg->crawl_stride_m;
    nav->pos.x += L * cosf(nav->yaw);
    nav->pos.y += L * sinf(nav->yaw);
    nav->dist_walked_m += L;
    nav->crawl_dist_m += L;
    /* Same error model as a walking stride, with a wider tolerance: the
     * crawl stride length is calibrated less often and varies more. */
    float sd = sqrtf(nav->var_crawl_m2) + cfg->crawl_stride_sigma * L;
    nav->var_crawl_m2 = sd * sd;

    float since = nav->n_crumbs ? dist2(nav->pos, nav->crumbs[nav->n_crumbs - 1].p) : 0.0f;
    prune_loops(nav, cfg);
    if (since >= cfg->breadcrumb_spacing_m) add_crumb(nav, cfg, t_ms);
    update_return_target(nav, cfg);
}

void ps_nav_tick(ps_nav_t *nav, const ps_config_t *cfg, uint32_t t_ms)
{
    if (nav->state == PS_NAV_IDLE) { nav->t_last_ms = t_ms; return; }
    /* Cadence lost: the next impact starts a new lock rather than extending
     * the old one. */
    if (nav->crawl_lock && t_ms - nav->t_last_crawl_stride_ms > cfg->crawl_max_interval_ms)
        nav->crawl_lock = 0;
    float dt = (float)(t_ms - nav->t_last_ms) * 0.001f;
    nav->t_last_ms = t_ms;
    if (dt <= 0.0f) return;

    if (t_ms - nav->t_last_imu_ms > cfg->imu_timeout_ms) {
        if (nav->state != PS_NAV_LOST) { nav->state = PS_NAV_LOST; nav->t_lost_ms = t_ms; nav->gaps++; }
    }

    /* Heading drift accumulates with time. */
    nav->heading_sigma_rad += cfg->gyro_drift_deg_per_min * DEG2RAD * dt / 60.0f;

    /*
     * Forget the walking speed once the steps stop. Without this the last
     * speed measured stayed on the books indefinitely, so an IMU outage that
     * began while the wearer was standing still coasted them 2.7 m across the
     * room — inventing exactly the kind of motion coasting exists to avoid
     * inventing. Only the step detector may set this; only the clock clears it.
     */
    if (nav->state == PS_NAV_TRACKING && nav->speed_mps > 0.0f &&
        t_ms - nav->t_last_step_ms > cfg->walk_idle_ms &&
        t_ms - nav->t_last_crawl_stride_ms > cfg->walk_idle_ms) {
        nav->speed_mps = 0.0f;
    }

    if (nav->state == PS_NAV_LOST) {
        /*
         * No IMU. Freezing the position here assumes the wearer stopped the
         * instant the sensor died, which is the one thing they certainly did
         * not do: they were walking, and they keep walking. Coast instead —
         * carry on at the speed and heading measured just before the loss.
         *
         * The estimate is a prediction, so the uncertainty still grows at the
         * full rate; but the mean is far better than a stop. Coasting is
         * capped, because after a few seconds "they kept going straight" is a
         * guess rather than an extrapolation.
         */
        float coast_s = (float)(t_ms - nav->t_lost_ms) * 0.001f;
        if (nav->speed_mps > 0.0f && coast_s <= cfg->imu_coast_max_s) {
            float v = nav->speed_mps * dt;
            nav->pos.x += v * cosf(nav->yaw);
            nav->pos.y += v * sinf(nav->yaw);
            nav->dist_walked_m += v;
            nav->gap_coast_m += v;
        }
        float sd = sqrtf(nav->var_gap_m2) + cfg->imu_gap_sigma_m_per_s * dt;
        nav->var_gap_m2 = sd * sd;
    } else if (nav->accel_base > cfg->crawl_accel_min &&
               t_ms - nav->t_last_step_ms > cfg->crawl_after_step_ms &&
               t_ms - nav->t_last_crawl_stride_ms > cfg->crawl_fallback_ms) {
        /* Moving, step detector quiet, and no crawl stride counted recently:
         * the gait is not readable (dragging a casualty, climbing, squeezing
         * through a gap). Fall back to the old assumed-speed model, which is
         * worse but never silently stops tracking. */
        /* Moving without counted steps: crawling (the standard posture in
         * heavy smoke), dragging a casualty, climbing. Advance at a nominal
         * crawl speed along the heading, and grow the uncertainty with the
         * full speed error since we cannot measure it. */
        nav->untracked_s += dt;
        float v = cfg->crawl_speed_mps * dt;
        nav->pos.x += v * cosf(nav->yaw);
        nav->pos.y += v * sinf(nav->yaw);
        nav->dist_walked_m += v;
        nav->crawl_dist_m += v;
        /*
         * This is the fallback: motion we can see but cannot measure, so the
         * displacement is a guess at a speed, not a count of strides.
         *
         * It goes in the GAP term, which does not cancel on the retrace.
         * Counted crawl strides do cancel — they share one stride-length
         * scale factor with the trail, exactly like walking — but shuffling
         * sideways along a wall shares nothing with the way back. Treating
         * the two alike made the device over-trust itself on routes where
         * nobody ever crawled, and the near-misses at the door went from one
         * to three.
         */
        float sd = sqrtf(nav->var_gap_m2) + cfg->crawl_speed_sigma * v;
        nav->var_gap_m2 = sd * sd;
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
    float sigma = sqrtf(nav->var_steps_m2 * frac + nav->var_crawl_m2 * frac + nav->var_gap_m2 +
                        bias * bias + cross * cross);
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
    g->exit_is_next = nav->return_target == 0;

    g->pos_sigma_m = nav->pos_sigma_m;
    g->confidence = nav->confidence;
}
