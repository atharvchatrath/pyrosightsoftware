/*
 * PyroSight host simulator.
 *
 * Runs the exact core library the ESP32-P4 firmware runs, against a simulated
 * firefighter walking through a smoke-filled floor plan:
 *   - a ray-cast thermal scene at the Lepton's 8.7 Hz (scene.c);
 *   - a BNO085 model at 100 Hz: game-rotation-vector yaw with gyro drift and
 *     noise, step detector with missed steps, linear acceleration;
 *   - a scripted search inbound, then a closed-loop walk OUT in which the
 *     simulated firefighter only follows the eyepiece arrow.
 *
 * It reports detection accuracy against ground truth, person distance error,
 * navigation error, whether the arrow alone led back to the door, the alerts
 * that were spoken, and per-stage processing time. Optional outputs: the
 * composited eyepiece frames (PPM), a CSV log, and a labelled dataset in the
 * format ml/dataset.py reads.
 *
 *   ps_sim [--scenario corridor|long|crawl|dropout] [--seed N] [--smoke 0..1]
 *          [--out DIR] [--every N] [--dump-dataset DIR] [--no-snap]
 *          [--step-bias F] [--quiet]
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "pyrosight/ps_system.h"
#include "scene.h"

#define PI_F 3.14159265f
#define DEG (PI_F / 180.0f)
#define IMU_DT_MS 10
#define FRAME_DT_MS 115 /* Lepton 3.5: ~8.7 Hz */

/* ------------------------------------------------------------ floor plan */
/* Door at the origin facing +x. Corridor A east, corridor B north, a room at
 * the end with a casualty and a fire. */
static const sim_wall_t WALLS[] = {
    { 0, -1, 15, -1 },  { 0, 1, 13, 1 },       /* corridor A */
    { 15, -1, 15, 12 }, { 13, 1, 13, 12 },     /* corridor B */
    { 9, 12, 13, 12 },  { 15, 12, 19, 12 },    /* room front wall with doorway */
    { 19, 12, 19, 20 }, { 9, 20, 19, 20 }, { 9, 12, 9, 20 },
};
static const sim_obj_t OBJS[] = {
    { OBJ_FIRE, 17.6f, 18.8f, 0, 1.1f, 1.3f, 0 },
    { OBJ_PERSON_LYING, 10.6f, 16.5f, 0.4f, 0, 0.35f, 0 },   /* casualty on the floor */
    { OBJ_PERSON_STANDING, 14.0f, 9.5f, 0, 0.5f, 1.75f, 0 }, /* crew member in corridor B */
    { OBJ_HOT_PIPE, 13.05f, 6.0f, 0, 2.0f, 0.12f, 2.1f },    /* hard negative: hot pipe */
};

typedef struct { float x, y; } wp_t;

typedef struct {
    const char *name;
    wp_t inbound[16];
    int n_inbound;
    float smoke;
    float layer_c;
    float gyro_drift_dps;    /* true gyro heading drift, deg/s */
    float step_miss;         /* probability a real step is not detected */
    float step_bias;         /* true stride = calibrated * (1 + bias) */
    int crawl_from, crawl_to;/* inbound leg indices crawled (no steps detected) */
    float cam_drop_s[2];     /* thermal camera outage window */
    float imu_drop_s[2];     /* IMU outage window */
    int laps;                /* repeat the search loop to accumulate drift */
} scenario_t;

static const scenario_t SCENARIOS[] = {
    { "corridor", { {7, 0}, {14, 0}, {14, 13.5f}, {11.5f, 15.5f}, {14, 16}, {16.5f, 16.5f} }, 6,
      0.5f, 110.0f, 0.02f, 0.03f, -0.06f, -1, -1, {0, 0}, {0, 0}, 1 },
    { "long", { {14, 0}, {14, 13.5f}, {11, 14}, {11, 18.5f}, {17, 18}, {17, 14}, {14, 13.5f}, {14, 4}, {14, 13.5f}, {11, 15} }, 10,
      0.6f, 120.0f, 0.06f, 0.05f, -0.12f, -1, -1, {0, 0}, {0, 0}, 3 },
    { "crawl", { {7, 0}, {14, 0}, {14, 13.5f}, {11.5f, 15.5f} }, 4,
      0.8f, 140.0f, 0.03f, 0.03f, -0.06f, 2, 3, {0, 0}, {0, 0}, 1 },
    { "dropout", { {7, 0}, {14, 0}, {14, 13.5f}, {11.5f, 15.5f}, {16.5f, 16.5f} }, 5,
      0.5f, 110.0f, 0.02f, 0.03f, -0.06f, -1, -1, {12, 15}, {30, 34}, 1 },
};

/* ------------------------------------------------------------- utilities */

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

typedef struct { double sum, max; double v[20000]; int n; } stat_t;
static void stat_add(stat_t *s, double v) { s->sum += v; if (v > s->max) s->max = v; if (s->n < 20000) s->v[s->n++] = v; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double stat_p95(stat_t *s) { if (!s->n) return 0; qsort(s->v, s->n, sizeof(double), cmpd); return s->v[(int)(s->n * 0.95)]; }

static void write_ppm(const char *path, const ps_fb_t *fb)
{
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", PS_DISP_W, PS_DISP_H);
    for (int i = 0; i < PS_DISP_W * PS_DISP_H; i++) {
        uint16_t c = fb->px[i];
        uint8_t rgb[3] = { (uint8_t)((c >> 8) & 0xF8), (uint8_t)((c >> 3) & 0xFC), (uint8_t)((c << 3) & 0xF8) };
        fwrite(rgb, 1, 3, f);
    }
    fclose(f);
}

static void mkdirs(const char *p)
{
    char buf[512];
    snprintf(buf, sizeof buf, "%s", p);
    for (char *s = buf + 1; *s; s++)
        if (*s == '/') { *s = 0; mkdir(buf, 0755); *s = '/'; }
    mkdir(buf, 0755);
}

static const char *phrase_line(const ps_alert_t *a, char *buf, size_t n)
{
    buf[0] = 0;
    for (int i = 0; i < a->n_parts; i++) {
        strncat(buf, ps_phrase_text[a->parts[i]], n - strlen(buf) - 1);
        if (i + 1 < a->n_parts) strncat(buf, " ", n - strlen(buf) - 1);
    }
    return buf;
}

/* ------------------------------------------------------------------- main */

typedef struct { int tp, fp, fn; } prf_t;

int main(int argc, char **argv)
{
    const char *scen_name = "corridor", *out_dir = NULL, *dump_dir = NULL;
    unsigned seed = 7;
    int every = 4;
    float smoke_override = -1, bias_override = -99;
    bool snap = true, quiet = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--scenario") && i + 1 < argc) scen_name = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--smoke") && i + 1 < argc) smoke_override = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--every") && i + 1 < argc) every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-dataset") && i + 1 < argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--step-bias") && i + 1 < argc) bias_override = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--no-snap")) snap = false;
        else if (!strcmp(argv[i], "--quiet")) quiet = true;
        else { fprintf(stderr, "usage: see header of sim/ps_sim.c\n"); return 2; }
    }
    const scenario_t *sc = NULL;
    for (size_t i = 0; i < sizeof SCENARIOS / sizeof *SCENARIOS; i++)
        if (!strcmp(SCENARIOS[i].name, scen_name)) sc = &SCENARIOS[i];
    if (!sc) { fprintf(stderr, "unknown scenario %s\n", scen_name); return 2; }

    sim_scene_seed(seed);
    sim_scene_t scene = {
        .walls = WALLS, .n_walls = sizeof WALLS / sizeof *WALLS,
        .objs = OBJS, .n_objs = sizeof OBJS / sizeof *OBJS,
        .wall_height = 2.6f, .ambient_c = 27.0f, .layer_c = sc->layer_c, .outside_c = 12.0f,
        .smoke = smoke_override >= 0 ? smoke_override : sc->smoke, .noise_c = 0.4f,
    };
    const float step_bias = bias_override > -98 ? bias_override : sc->step_bias;

    static ps_system_t sys;
    ps_config_t cfg;
    ps_config_default(&cfg);
    cfg.heading_snap = snap;
    ps_system_init(&sys, &cfg, 0);
    ps_system_on_battery(&sys, 3950, 0);

    FILE *log = NULL;
    if (out_dir) {
        char p[600];
        snprintf(p, sizeof p, "%s/frames", out_dir);
        mkdirs(p);
        snprintf(p, sizeof p, "%s/log.csv", out_dir);
        log = fopen(p, "w");
        if (log) fprintf(log, "t_s,phase,true_x,true_y,true_yaw_deg,est_x,est_y,est_yaw_deg,pos_err_m,sigma_m,confidence,level,route_bearing_deg,route_dist_m,crumbs,dets\n");
    }
    FILE *meta = NULL;
    if (dump_dir) {
        char p[600];
        snprintf(p, sizeof p, "%s/images", dump_dir); mkdirs(p);
        snprintf(p, sizeof p, "%s/labels", dump_dir); mkdirs(p);
        snprintf(p, sizeof p, "%s/meta.csv", dump_dir);
        meta = fopen(p, "w");
        if (meta) fprintf(meta, "id,smoke\n");
    }

    /* True state of the simulated firefighter. */
    float tx = 0.3f, ty = 0.0f, tyaw = 0.0f;
    const float cal_step = cfg.step_length_m;
    float gyro_err = 0.0f;           /* accumulated heading error, rad */
    float yaw_world_offset = 1.1f;   /* IMU yaw is relative to an arbitrary reference */
    int phase = 0;                   /* 0 inbound, 1 scan, 2 outbound */
    int leg = 0, lap = 0;
    float step_phase = 0, scan_turned = 0, slide_yaw = 0;
    int slide_steps = 0;
    uint32_t t = 0, next_frame = 0, out_frame = 0, frame_id = 0;
    uint32_t t_out_start = 0, t_limit = 600000;
    bool hose_used = false, exited = false;
    int dataset_n = 0;

    static ps_thermal_frame_t raw;
    static ps_fb_t fb;
    sim_truth_t truth;
    stat_t st_pipe = {0}, st_det = {0}, st_render = {0};
    prf_t prf[PS_NUM_CLASSES] = {{0}};
    double dist_abs_err = 0, dist_rel_err = 0;
    int dist_n = 0;
    float max_pos_err = 0, min_conf = 1;
    int alerts_spoken = 0;
    char line[160];

    /* At the doorway: mark entry. */
    ps_system_on_yaw(&sys, ps_wrap_pi(tyaw + yaw_world_offset), t);
    ps_system_on_button(&sys, PS_BTN_MARK_ENTRY, t);

    for (; t < t_limit; t += IMU_DT_MS) {
        const float ts = t / 1000.0f;
        const bool imu_down = ts >= sc->imu_drop_s[0] && ts < sc->imu_drop_s[1];
        const bool cam_down = ts >= sc->cam_drop_s[0] && ts < sc->cam_drop_s[1];

        /* ---- decide where the firefighter wants to go ---- */
        float desired = tyaw;
        bool want_walk = false, crawling = false;
        if (phase == 0) {
            wp_t w = sc->inbound[leg];
            float dx = w.x - tx, dy = w.y - ty;
            if (sqrtf(dx * dx + dy * dy) < 0.5f) {
                leg++;
                if (leg >= sc->n_inbound) {
                    if (++lap < sc->laps) leg = 2; /* search the room again */
                    else { phase = 1; scan_turned = 0; }
                }
            } else {
                desired = atan2f(dy, dx);
                want_walk = true;
                crawling = leg >= sc->crawl_from && leg <= sc->crawl_to && sc->crawl_from >= 0;
            }
        } else if (phase == 1) {
            /* Sweep the room with the camera, then ask the device for the way out. */
            desired = tyaw + 60 * DEG;
            scan_turned += 60 * DEG * IMU_DT_MS / 1000.0f;
            if (scan_turned > 2 * PI_F) {
                phase = 2;
                t_out_start = t;
                ps_system_on_button(&sys, PS_BTN_WHERE_OUT, t);
            }
        } else {
            /* Outbound: follow ONLY the eyepiece arrow. If the device says the
             * estimate is unreliable, the firefighter follows the hose (the
             * true inbound path) instead, which is what the alert tells them. */
            ps_nav_guidance_t g;
            ps_nav_guidance(&sys.nav, &g);
            if (sys.alerts.level == PS_NAVCONF_UNRELIABLE && !hose_used) {
                /* Find the hose: the nearest point of the inbound route. */
                hose_used = true;
                float best = 1e9f;
                for (int k = 0; k < sc->n_inbound; k++) {
                    float d = hypotf(sc->inbound[k].x - tx, sc->inbound[k].y - ty);
                    if (d < best) { best = d; leg = k; }
                }
            }
            if (hose_used) {
                int k = leg < sc->n_inbound ? leg : sc->n_inbound - 1;
                wp_t w = k >= 0 ? sc->inbound[k] : (wp_t){ 0, 0 };
                if (sqrtf((w.x - tx) * (w.x - tx) + (w.y - ty) * (w.y - ty)) < 0.6f) leg--;
                if (leg < 0) w = (wp_t){ -0.5f, 0 };
                desired = atan2f(w.y - ty, w.x - tx);
                want_walk = true;
            } else if (g.valid && g.route_dist_m > 0.6f) {
                desired = tyaw + g.route_bearing_rel_deg * DEG;
                want_walk = true;
            }
            if (tx < 0.6f && fabsf(ty) < 1.0f) { exited = true; break; }
            if (g.valid && g.route_dist_m <= 0.6f && !hose_used) {
                /* The device says "EXIT" here. Real firefighters would now feel
                 * for the door; stop and measure how far off we are. */
                break;
            }
        }
        if (phase == 2 && leg >= sc->n_inbound) leg = sc->n_inbound - 1;

        /* A blocked firefighter feels along the wall instead of pushing into it. */
        const float goal = desired;
        if (slide_steps > 0) desired = slide_yaw;

        /* ---- move: turn at up to 120 deg/s, walk when roughly aligned ---- */
        float err = ps_wrap_pi(desired - tyaw);
        float max_turn = 120 * DEG * IMU_DT_MS / 1000.0f;
        float dyaw = err > max_turn ? max_turn : (err < -max_turn ? -max_turn : err);
        tyaw = ps_wrap_pi(tyaw + dyaw);
        bool step_now = false;
        float accel = 0.2f + 0.05f * sim_randn();
        if (want_walk && fabsf(err) < 25 * DEG) {
            if (crawling) {
                float v = 0.35f * IMU_DT_MS / 1000.0f; /* crawl, no step events */
                float nx = tx + v * cosf(tyaw), ny = ty + v * sinf(tyaw);
                sim_collide(&scene, tx, ty, &nx, &ny);
                tx = nx; ty = ny;
                accel = 1.8f + 0.3f * sim_randn();
            } else {
                accel = 2.2f + 0.4f * sim_randn();
                step_phase += 1.6f * IMU_DT_MS / 1000.0f; /* 1.6 steps/s in smoke */
                if (step_phase >= 1.0f) {
                    step_phase -= 1.0f;
                    float L = cal_step * (1 + step_bias) * (1 + 0.05f * sim_randn());
                    float nx = tx + L * cosf(tyaw), ny = ty + L * sinf(tyaw);
                    bool hit = sim_collide(&scene, tx, ty, &nx, &ny);
                    float moved = sqrtf((nx - tx) * (nx - tx) + (ny - ty) * (ny - ty));
                    if (hit && moved < 0.5f * L && slide_steps == 0) {
                        /* Blocked: stop, turn along the wall toward the side the
                         * arrow favours, and follow it for a while. */
                        if (nx == tx) slide_yaw = (sinf(goal) > 0.15f || (fabsf(sinf(goal)) <= 0.15f && sim_rand01() < 0.5f)) ? PI_F / 2 : -PI_F / 2;
                        else slide_yaw = (cosf(goal) > 0.15f || (fabsf(cosf(goal)) <= 0.15f && sim_rand01() < 0.5f)) ? 0 : PI_F;
                        slide_steps = 14;
                    } else {
                        tx = nx; ty = ny;
                        /* Shuffling against a wall is not a counted stride. */
                        step_now = moved > 0.3f * L && sim_rand01() >= sc->step_miss;
                        if (slide_steps > 0) {
                            slide_steps--;
                            float px2 = tx + 1.0f * cosf(goal), py2 = ty + 1.0f * sinf(goal);
                            if (!sim_collide(&scene, tx, ty, &px2, &py2)) slide_steps = 0; /* opening found */
                        }
                    }
                }
            }
        }

        /* ---- BNO085 model ---- */
        gyro_err += (sc->gyro_drift_dps * DEG + 0.002f * sim_randn()) * IMU_DT_MS / 1000.0f;
        gyro_err += 0.01f * dyaw; /* 1% gyro scale-factor error on turns */
        if (!imu_down) {
            float meas = ps_wrap_pi(tyaw + yaw_world_offset + gyro_err + 0.003f * sim_randn());
            ps_system_on_yaw(&sys, meas, t);
            ps_system_on_linear_accel(&sys, fabsf(accel), t);
            if (step_now) ps_system_on_step(&sys, t);
        }

        /* ---- Lepton frame ---- */
        if (t >= next_frame) {
            next_frame += FRAME_DT_MS;
            frame_id++;
            sim_render(&scene, tx, ty, tyaw, 1.5f, ts, &raw, &truth);
            raw.frame_id = frame_id;
            raw.t_ms = t;
            if (!cam_down) {
                double t0 = now_us();
                ps_system_on_frame(&sys, &raw);
                double t1 = now_us();
                ps_system_run_classical(&sys);
                double t2 = now_us();
                stat_add(&st_pipe, t1 - t0);
                stat_add(&st_det, t2 - t1);

                /* Score detections against ground truth (IoU >= 0.3). */
                bool used[PS_MAX_DETECTIONS] = {0};
                for (int gi = 0; gi < truth.n; gi++) {
                    const ps_detection_t *gt = &truth.gt[gi];
                    if (gt->w * gt->h < 12) continue; /* too small to expect */
                    int best = -1;
                    float biou = 0.3f;
                    for (int di = 0; di < sys.dets.n; di++) {
                        if (used[di] || sys.dets.d[di].cls != gt->cls) continue;
                        float iou = ps_iou(&sys.dets.d[di], gt);
                        if (iou >= biou) { biou = iou; best = di; }
                    }
                    if (best >= 0) {
                        used[best] = true;
                        prf[gt->cls].tp++;
                        const ps_detection_t *d = &sys.dets.d[best];
                        if (gt->cls == PS_CLASS_PERSON && d->dist_m > 0 && !d->truncated) {
                            dist_abs_err += fabs(d->dist_m - gt->dist_m);
                            dist_rel_err += fabs(d->dist_m - gt->dist_m) / gt->dist_m;
                            dist_n++;
                        }
                    } else {
                        prf[gt->cls].fn++;
                    }
                }
                for (int di = 0; di < sys.dets.n; di++) if (!used[di]) prf[sys.dets.d[di].cls].fp++;
            }
            if (dump_dir && frame_id % 3 == 0) {
                char p[700];
                snprintf(p, sizeof p, "%s/images/%06d.bin", dump_dir, dataset_n);
                FILE *f = fopen(p, "wb");
                if (f) { fwrite(raw.px, sizeof(ps_dc_t), PS_THERM_PIXELS, f); fclose(f); }
                snprintf(p, sizeof p, "%s/labels/%06d.txt", dump_dir, dataset_n);
                f = fopen(p, "w");
                if (f) {
                    for (int gi = 0; gi < truth.n; gi++) {
                        const ps_detection_t *g = &truth.gt[gi];
                        fprintf(f, "%d %.1f %.1f %.1f %.1f %.2f\n", g->cls, g->x, g->y, g->w, g->h,
                                g->cls == PS_CLASS_PERSON ? g->dist_m : 0.0f);
                    }
                    fclose(f);
                }
                if (meta) fprintf(meta, "%d,%.2f\n", dataset_n, scene.smoke);
                dataset_n++;
            }
            double t3 = now_us();
            ps_system_render(&sys, &fb, t);
            stat_add(&st_render, now_us() - t3);
            if (out_dir && frame_id % (uint32_t)every == 0) {
                char p[700];
                snprintf(p, sizeof p, "%s/frames/frame_%05u.ppm", out_dir, out_frame++);
                write_ppm(p, &fb);
            }
        }

        ps_system_tick(&sys, t);
        ps_alert_t al;
        while (ps_system_next_alert(&sys, &al)) {
            alerts_spoken++;
            if (!quiet) printf("  [%6.1fs] AUDIO: %s\n", ts, phrase_line(&al, line, sizeof line));
        }

        /* ---- navigation error bookkeeping (entry frame: +x = entry heading) ---- */
        float ex = sys.nav.pos.x, ey = sys.nav.pos.y;
        float perr = sqrtf((ex - (tx - 0.3f)) * (ex - (tx - 0.3f)) + (ey - ty) * (ey - ty));
        if (perr > max_pos_err) max_pos_err = perr;
        if (sys.nav.confidence < min_conf) min_conf = sys.nav.confidence;
        if (log && t % 200 == 0) {
            ps_nav_guidance_t g;
            ps_nav_guidance(&sys.nav, &g);
            fprintf(log, "%.2f,%d,%.3f,%.3f,%.1f,%.3f,%.3f,%.1f,%.3f,%.3f,%.3f,%d,%.1f,%.2f,%d,%d\n",
                    ts, phase, tx - 0.3f, ty, tyaw / DEG, ex, ey, sys.nav.yaw / DEG, perr, g.pos_sigma_m,
                    g.confidence, sys.alerts.level, g.route_bearing_rel_deg, g.route_dist_m, sys.nav.n_crumbs, sys.dets.n);
        }
    }
    if (log) fclose(log);
    if (meta) fclose(meta);

    /* ------------------------------------------------------------ report */
    float door_err = sqrtf((tx - 0.0f) * (tx - 0.0f) + ty * ty);
    char summary[4096];
    int o = 0;
    o += snprintf(summary + o, sizeof summary - o, "scenario            %s (seed %u, smoke %.2f, step bias %+.0f%%, heading snap %s)\n",
                  sc->name, seed, scene.smoke, step_bias * 100, snap ? "on" : "off");
    o += snprintf(summary + o, sizeof summary - o, "simulated time      %.1f s, inbound path %.1f m, %u frames\n",
                  t / 1000.0f, sys.nav.dist_walked_m, frame_id);
    const char *names[2] = { "fire", "person" };
    for (int c = 0; c < PS_NUM_CLASSES; c++) {
        int tp = prf[c].tp, fp = prf[c].fp, fn = prf[c].fn;
        o += snprintf(summary + o, sizeof summary - o, "detector %-7s    precision %.2f recall %.2f (tp %d fp %d fn %d) [threshold detector]\n",
                      names[c], tp + fp ? (double)tp / (tp + fp) : 0, tp + fn ? (double)tp / (tp + fn) : 0, tp, fp, fn);
    }
    o += snprintf(summary + o, sizeof summary - o, "person distance     mean abs error %.2f m, mean rel error %.0f%% (%d matches)\n",
                  dist_n ? dist_abs_err / dist_n : 0, dist_n ? 100 * dist_rel_err / dist_n : 0, dist_n);
    o += snprintf(summary + o, sizeof summary - o, "navigation          max position error %.2f m, lowest confidence %.2f, crumbs left %d\n",
                  max_pos_err, min_conf, sys.nav.n_crumbs);
    o += snprintf(summary + o, sizeof summary - o, "way out             %s; walk-out took %.1f s; ended %.2f m from the door\n",
                  exited ? (hose_used ? "reached door (switched to hose after unreliable warning)" : "reached door following the arrow")
                         : (hose_used ? "did NOT reach door (hose)" : "device said EXIT"),
                  (t - t_out_start) / 1000.0f, door_err);
    o += snprintf(summary + o, sizeof summary - o, "alerts spoken       %d\n", alerts_spoken);
    o += snprintf(summary + o, sizeof summary - o, "host timing (us)    pipeline mean %.0f p95 %.0f | detector mean %.0f p95 %.0f | render mean %.0f p95 %.0f\n",
                  st_pipe.n ? st_pipe.sum / st_pipe.n : 0, stat_p95(&st_pipe), st_det.n ? st_det.sum / st_det.n : 0,
                  stat_p95(&st_det), st_render.n ? st_render.sum / st_render.n : 0, stat_p95(&st_render));
    o += snprintf(summary + o, sizeof summary - o, "health              camera losses %u, IMU losses %u, frames dropped %u\n",
                  sys.stats.camera_losses, sys.stats.imu_losses, sys.stats.frames_dropped);
    if (dump_dir) o += snprintf(summary + o, sizeof summary - o, "dataset             %d labelled frames -> %s\n", dataset_n, dump_dir);
    printf("%s", summary);
    if (out_dir) {
        char p[600];
        snprintf(p, sizeof p, "%s/summary.txt", out_dir);
        FILE *f = fopen(p, "w");
        if (f) { fputs(summary, f); fclose(f); }
    }
    return 0;
}
