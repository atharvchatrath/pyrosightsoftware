#include "sim_world.h"

/* Simulated crawl gait: mean speed and distance per hand-knee cycle. */
#define CRAWL_SPEED_MPS 0.35f
#define CRAWL_STRIDE_M 0.45f

#include <math.h>
#include <string.h>

#define PI_F 3.14159265f
#define DEG (PI_F / 180.0f)
#define START_X 0.3f

/* Door at the origin facing +x. Corridor A east, corridor B north, a room at
 * the end with a casualty and a fire. */
const sim_wall_t SIM_WALLS[] = {
    { 0, -1, 15, -1 },  { 0, 1, 13, 1 },       /* corridor A */
    { 15, -1, 15, 12 }, { 13, 1, 13, 12 },     /* corridor B */
    { 9, 12, 13, 12 },  { 15, 12, 19, 12 },    /* room front wall with doorway */
    { 19, 12, 19, 20 }, { 9, 20, 19, 20 }, { 9, 12, 9, 20 },
};
const int SIM_N_WALLS = sizeof SIM_WALLS / sizeof *SIM_WALLS;

const sim_obj_t SIM_OBJS[] = {
    { OBJ_FIRE, 17.6f, 18.8f, 0, 1.1f, 1.3f, 0 },
    { OBJ_PERSON_LYING, 10.6f, 16.5f, 0.4f, 0, 0.35f, 0 },   /* casualty on the floor */
    { OBJ_PERSON_STANDING, 14.0f, 9.5f, 0, 0.5f, 1.75f, 0 }, /* crew member in corridor B */
    { OBJ_HOT_PIPE, 13.05f, 6.0f, 0, 2.0f, 0.12f, 2.1f },    /* hard negative: hot pipe */
};
const int SIM_N_OBJS = sizeof SIM_OBJS / sizeof *SIM_OBJS;

const sim_scenario_t SIM_SCENARIOS[] = {
    { "corridor", { {7, 0}, {14, 0}, {14, 13.5f}, {11.5f, 15.5f}, {14, 16}, {16.5f, 16.5f} }, 6,
      0.5f, 110.0f, 0.02f, 0.03f, -0.06f, -1, -1, {0, 0}, {0, 0}, 1 },
    { "long", { {14, 0}, {14, 13.5f}, {11, 14}, {11, 18.5f}, {17, 18}, {17, 14}, {14, 13.5f}, {14, 4}, {14, 13.5f}, {11, 15} }, 10,
      0.6f, 120.0f, 0.06f, 0.05f, -0.12f, -1, -1, {0, 0}, {0, 0}, 3 },
    { "crawl", { {7, 0}, {14, 0}, {14, 13.5f}, {11.5f, 15.5f} }, 4,
      0.8f, 140.0f, 0.03f, 0.03f, -0.06f, 2, 3, {0, 0}, {0, 0}, 1 },
    { "dropout", { {7, 0}, {14, 0}, {14, 13.5f}, {11.5f, 15.5f}, {16.5f, 16.5f} }, 5,
      0.5f, 110.0f, 0.02f, 0.03f, -0.06f, -1, -1, {12, 15}, {30, 34}, 1 },
};
const int SIM_N_SCENARIOS = sizeof SIM_SCENARIOS / sizeof *SIM_SCENARIOS;

bool sim_world_init(sim_world_t *w, ps_system_t *sys, const char *scenario, unsigned seed,
                    float smoke, float step_bias, bool heading_snap)
{
    const sim_scenario_t *sc = NULL;
    for (int i = 0; i < SIM_N_SCENARIOS; i++)
        if (!strcmp(SIM_SCENARIOS[i].name, scenario)) sc = &SIM_SCENARIOS[i];
    if (!sc) return false;

    memset(w, 0, sizeof(*w));
    w->sc = sc;
    w->sys = sys;
    w->seed = seed;
    sim_scene_seed(seed);
    w->scene = (sim_scene_t){
        .walls = SIM_WALLS, .n_walls = SIM_N_WALLS, .objs = SIM_OBJS, .n_objs = SIM_N_OBJS,
        .wall_height = 2.6f, .ambient_c = 27.0f, .layer_c = sc->layer_c, .outside_c = 12.0f,
        .smoke = smoke >= 0 ? smoke : sc->smoke, .noise_c = 0.4f,
    };
    w->step_bias = step_bias > -98 ? step_bias : sc->step_bias;

    ps_config_t cfg;
    ps_config_default(&cfg);
    cfg.heading_snap = heading_snap;
    ps_system_init(sys, &cfg, 0);
    ps_system_on_battery(sys, 3950, 0);

    w->tx = START_X;
    w->yaw_world_offset = 1.1f; /* IMU yaw is relative to an arbitrary reference */

    /* At the doorway: mark entry. */
    ps_system_on_yaw(sys, ps_wrap_pi(w->tyaw + w->yaw_world_offset), 0);
    ps_system_on_button(sys, PS_BTN_MARK_ENTRY, 0);
    return true;
}

void sim_world_estimate(const sim_world_t *w, float *x, float *y)
{
    /* Entry frame: origin where the entry was marked, +x the heading then (0). */
    *x = w->sys->nav.pos.x + START_X;
    *y = w->sys->nav.pos.y;
}

void sim_world_truth_detections(const sim_world_t *w, ps_detections_t *out)
{
    memset(out, 0, sizeof(*out));
    for (int i = 0; i < w->truth.n && out->n < PS_MAX_DETECTIONS; i++) {
        out->d[out->n] = w->truth.gt[i];
        out->d[out->n].dist_m = 0; /* the device estimates distance itself */
        out->n++;
    }
    out->frame_id = w->raw.frame_id;
    out->t_ms = w->raw.t_ms;
    out->source = PS_DETECTOR_REFERENCE;
}

int sim_world_step(sim_world_t *w)
{
    if (w->done) return SIM_EV_DONE;
    const sim_scenario_t *sc = w->sc;
    ps_system_t *sys = w->sys;
    const uint32_t t = w->t;
    const float ts = t / 1000.0f;
    const bool imu_down = ts >= sc->imu_drop_s[0] && ts < sc->imu_drop_s[1];
    w->cam_down = ts >= sc->cam_drop_s[0] && ts < sc->cam_drop_s[1];
    int ev = SIM_EV_NONE;

    /* ---- decide where the firefighter wants to go ---- */
    float desired = w->tyaw;
    bool want_walk = false, crawling = false;
    if (w->phase == SIM_PHASE_INBOUND) {
        sim_wp_t p = sc->inbound[w->leg];
        float dx = p.x - w->tx, dy = p.y - w->ty;
        if (sqrtf(dx * dx + dy * dy) < 0.5f) {
            w->leg++;
            if (w->leg >= sc->n_inbound) {
                if (++w->lap < sc->laps) w->leg = 2; /* search the room again */
                else { w->phase = SIM_PHASE_SCAN; w->scan_turned = 0; }
            }
        } else {
            desired = atan2f(dy, dx);
            want_walk = true;
            crawling = sc->crawl_from >= 0 && w->leg >= sc->crawl_from && w->leg <= sc->crawl_to;
        }
    } else if (w->phase == SIM_PHASE_SCAN) {
        /* Sweep the room with the camera, then ask the device for the way out. */
        desired = w->tyaw + 60 * DEG;
        w->scan_turned += 60 * DEG * SIM_IMU_DT_MS / 1000.0f;
        if (w->scan_turned > 2 * PI_F) {
            w->phase = SIM_PHASE_OUTBOUND;
            w->t_out_start = t;
            ps_system_on_button(sys, PS_BTN_WHERE_OUT, t);
        }
    } else {
        /* Outbound: follow ONLY the eyepiece arrow. If the device says the
         * estimate is unreliable, the firefighter follows the hose (the true
         * inbound path) instead, which is what the alert tells them. */
        ps_nav_guidance_t g;
        ps_nav_guidance(&sys->nav, &g);
        if (sys->alerts.level == PS_NAVCONF_UNRELIABLE && !w->hose_used) {
            /* Find the hose: the nearest point of the inbound route. */
            w->hose_used = true;
            float best = 1e9f;
            for (int k = 0; k < sc->n_inbound; k++) {
                float d = hypotf(sc->inbound[k].x - w->tx, sc->inbound[k].y - w->ty);
                if (d < best) { best = d; w->leg = k; }
            }
        }
        if (w->hose_used) {
            int k = w->leg < sc->n_inbound ? w->leg : sc->n_inbound - 1;
            sim_wp_t p = k >= 0 ? sc->inbound[k] : (sim_wp_t){ 0, 0 };
            if (hypotf(p.x - w->tx, p.y - w->ty) < 0.6f) w->leg--;
            if (w->leg < 0) p = (sim_wp_t){ -0.5f, 0 };
            desired = atan2f(p.y - w->ty, p.x - w->tx);
            want_walk = true;
        } else if (g.valid && g.route_dist_m > 0.6f) {
            desired = w->tyaw + g.route_bearing_rel_deg * DEG;
            want_walk = true;
        }
        if (w->tx < 0.6f && fabsf(w->ty) < 1.0f) { w->exited = true; w->done = true; return SIM_EV_DONE; }
        if (g.valid && g.route_dist_m <= 0.6f && !w->hose_used) {
            /* The device says "EXIT" here. A real firefighter would now feel
             * for the door; stop and measure how far off we are. */
            w->done = true;
            return SIM_EV_DONE;
        }
        if (w->leg >= sc->n_inbound) w->leg = sc->n_inbound - 1;
    }
    if (t >= 600000) { w->done = true; return SIM_EV_DONE; }

    /* A blocked firefighter feels along the wall instead of pushing into it. */
    const float goal = desired;
    if (w->slide_steps > 0) desired = w->slide_yaw;

    /* ---- move: turn at up to 120 deg/s, walk when roughly aligned ---- */
    float err = ps_wrap_pi(desired - w->tyaw);
    float max_turn = 120 * DEG * SIM_IMU_DT_MS / 1000.0f;
    float dyaw = err > max_turn ? max_turn : (err < -max_turn ? -max_turn : err);
    w->tyaw = ps_wrap_pi(w->tyaw + dyaw);
    bool step_now = false;
    float accel = 0.2f + 0.05f * sim_randn();
    if (want_walk && fabsf(err) < 25 * DEG) {
        if (crawling) {
            /*
             * Crawling on hands and knees. The step detector does not fire
             * (the BNO085 looks for a walking gait), but the motion is not
             * smooth: each hand-knee cycle is an impact the accelerometer
             * sees clearly. Modelling crawl as constant acceleration made the
             * signal featureless and left the device nothing to measure but
             * elapsed time, which is not what a real IMU gives you.
             *
             * ~0.78 cycles/s advancing ~0.45 m each = the same 0.35 m/s mean
             * as before, so inbound travel is unchanged; what is new is the
             * periodic structure a stride counter can lock onto.
             */
            float v = CRAWL_SPEED_MPS * SIM_IMU_DT_MS / 1000.0f;
            float nx = w->tx + v * cosf(w->tyaw), ny = w->ty + v * sinf(w->tyaw);
            sim_collide(&w->scene, w->tx, w->ty, &nx, &ny);
            w->tx = nx; w->ty = ny;
            w->crawl_phase += (CRAWL_SPEED_MPS / CRAWL_STRIDE_M) * SIM_IMU_DT_MS / 1000.0f;
            float impact = 0.0f;
            if (w->crawl_phase >= 1.0f) { w->crawl_phase -= 1.0f; impact = 1.0f; }
            /* The impact decays over the ~120 ms after the hand/knee lands. */
            float tail = expf(-w->crawl_phase * CRAWL_STRIDE_M / CRAWL_SPEED_MPS / 0.12f);
            accel = 1.1f + 2.6f * (impact > 0.0f ? 1.0f : tail) + 0.25f * sim_randn();
        } else {
            accel = 2.2f + 0.4f * sim_randn();
            w->step_phase += 1.6f * SIM_IMU_DT_MS / 1000.0f; /* 1.6 steps/s in smoke */
            if (w->step_phase >= 1.0f) {
                w->step_phase -= 1.0f;
                float L = sys->cfg.step_length_m * (1 + w->step_bias) * (1 + 0.05f * sim_randn());
                float nx = w->tx + L * cosf(w->tyaw), ny = w->ty + L * sinf(w->tyaw);
                bool hit = sim_collide(&w->scene, w->tx, w->ty, &nx, &ny);
                float moved = hypotf(nx - w->tx, ny - w->ty);
                if (hit && moved < 0.5f * L && w->slide_steps == 0) {
                    /* Blocked: stop, turn along the wall toward the side the
                     * arrow favours, and follow it for a while. */
                    if (nx == w->tx) w->slide_yaw = (sinf(goal) > 0.15f || (fabsf(sinf(goal)) <= 0.15f && sim_rand01() < 0.5f)) ? PI_F / 2 : -PI_F / 2;
                    else w->slide_yaw = (cosf(goal) > 0.15f || (fabsf(cosf(goal)) <= 0.15f && sim_rand01() < 0.5f)) ? 0 : PI_F;
                    w->slide_steps = 14;
                } else {
                    w->tx = nx; w->ty = ny;
                    /* Shuffling against a wall is not a counted stride. */
                    step_now = moved > 0.3f * L && sim_rand01() >= sc->step_miss;
                    if (w->slide_steps > 0) {
                        w->slide_steps--;
                        float px2 = w->tx + 1.0f * cosf(goal), py2 = w->ty + 1.0f * sinf(goal);
                        if (!sim_collide(&w->scene, w->tx, w->ty, &px2, &py2)) w->slide_steps = 0; /* opening found */
                    }
                }
            }
        }
    }

    /* ---- BNO085 model ---- */
    w->gyro_err += (sc->gyro_drift_dps * DEG + 0.002f * sim_randn()) * SIM_IMU_DT_MS / 1000.0f;
    w->gyro_err += 0.01f * dyaw; /* 1% gyro scale-factor error on turns */
    if (!imu_down) {
        float meas = ps_wrap_pi(w->tyaw + w->yaw_world_offset + w->gyro_err + 0.003f * sim_randn());
        ps_system_on_yaw(sys, meas, t);
        ps_system_on_linear_accel(sys, fabsf(accel), t);
        if (step_now) ps_system_on_step(sys, t);
    }

    /* ---- Lepton frame ---- */
    if (t >= w->next_frame) {
        w->next_frame += SIM_FRAME_DT_MS;
        w->frame_id++;
        sim_render(&w->scene, w->tx, w->ty, w->tyaw, sys->cfg.camera_height_m, ts, &w->raw, &w->truth);
        w->raw.frame_id = w->frame_id;
        w->raw.t_ms = t;
        if (!w->cam_down) {
            ps_system_on_frame(sys, &w->raw);
            ev = SIM_EV_FRAME;
        }
    }

    ps_system_tick(sys, t);
    w->t += SIM_IMU_DT_MS;
    return ev;
}
