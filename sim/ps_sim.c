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
 *          [--step-bias F] [--model FILE.psnn] [--quiet]
 *
 * Detector: the threshold detector by default; --model runs the trained
 * network (ml/export_nnref.py output) through the float reference runtime.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>


#include "nn_ref.h"
#include "sim_world.h"

#define DEG (3.14159265f / 180.0f)

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
    const char *scen_name = "corridor", *out_dir = NULL, *dump_dir = NULL, *model_path = NULL;
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
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--quiet")) quiet = true;
        else { fprintf(stderr, "usage: see header of sim/ps_sim.c\n"); return 2; }
    }
    nn_ref_t *model = NULL;
    static uint8_t *model_blob;
    if (model_path) {
        FILE *f = fopen(model_path, "rb");
        long n = -1;
        if (f && !fseek(f, 0, SEEK_END)) { n = ftell(f); rewind(f); }
        model_blob = n > 0 ? malloc((size_t)n) : NULL;
        if (model_blob && fread(model_blob, 1, (size_t)n, f) == (size_t)n) model = nn_ref_load(model_blob, (size_t)n);
        if (f) fclose(f);
        if (!model) { fprintf(stderr, "cannot load model %s\n", model_path); return 2; }
    }
    static float nn_heat[2400], nn_wh[2400], nn_off[2400];
    static ps_system_t sys;
    static sim_world_t world;
    sim_world_t *w = &world;
    if (!sim_world_init(w, &sys, scen_name, seed, smoke_override, bias_override, snap)) {
        fprintf(stderr, "unknown scenario %s\n", scen_name);
        return 2;
    }
    const sim_scenario_t *sc = w->sc;
    const float step_bias = w->step_bias;

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

    uint32_t out_frame = 0;
    int dataset_n = 0;
    static ps_fb_t fb;
    stat_t st_pipe = {0}, st_det = {0}, st_render = {0};
    prf_t prf[PS_NUM_CLASSES] = {{0}};
    double dist_abs_err = 0, dist_rel_err = 0;
    int dist_n = 0;
    float max_pos_err = 0, min_conf = 1;
    int alerts_spoken = 0;
    char line[160];

    for (;;) {
        const uint32_t t = w->t;
        const float ts = t / 1000.0f;
        const uint32_t frame_before = w->frame_id;
        double t0 = now_us();
        int ev = sim_world_step(w);
        if (ev == SIM_EV_DONE) break;
        if (ev == SIM_EV_FRAME) {
            double t1 = now_us();
            if (model) {
                ps_detections_t d;
                memset(&d, 0, sizeof d);
                if (nn_ref_run(model, sys.model_input.px, nn_heat, nn_wh, nn_off) == 0) {
                    ps_centernet_decode(nn_heat, nn_wh, nn_off, 40, 30, 4, sys.cfg.score_threshold, &d);
                    ps_nms(&d, sys.cfg.nms_iou);
                }
                d.frame_id = sys.frame.frame_id;
                d.t_ms = sys.frame.t_ms;
                ps_system_on_detections(&sys, &d, t);
            } else {
                ps_system_run_classical(&sys);
            }
            double t2 = now_us();
            stat_add(&st_pipe, t1 - t0); /* includes scene rendering on the host */
            stat_add(&st_det, t2 - t1);

            /* Score detections against ground truth (IoU >= 0.3). */
            const sim_truth_t *truth = &w->truth;
            bool used[PS_MAX_DETECTIONS] = {0};
            for (int gi = 0; gi < truth->n; gi++) {
                const ps_detection_t *gt = &truth->gt[gi];
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
        if (w->frame_id != frame_before) {
            if (dump_dir && w->frame_id % 3 == 0) {
                char p[700];
                snprintf(p, sizeof p, "%s/images/%06d.bin", dump_dir, dataset_n);
                FILE *f = fopen(p, "wb");
                if (f) { fwrite(w->raw.px, sizeof(ps_dc_t), PS_THERM_PIXELS, f); fclose(f); }
                snprintf(p, sizeof p, "%s/labels/%06d.txt", dump_dir, dataset_n);
                f = fopen(p, "w");
                if (f) {
                    for (int gi = 0; gi < w->truth.n; gi++) {
                        const ps_detection_t *g = &w->truth.gt[gi];
                        fprintf(f, "%d %.1f %.1f %.1f %.1f %.2f\n", g->cls, g->x, g->y, g->w, g->h,
                                g->cls == PS_CLASS_PERSON ? g->dist_m : 0.0f);
                    }
                    fclose(f);
                }
                if (meta) fprintf(meta, "%d,%.2f\n", dataset_n, w->scene.smoke);
                dataset_n++;
            }
            double t3 = now_us();
            ps_system_render(&sys, &fb, t);
            stat_add(&st_render, now_us() - t3);
            if (out_dir && w->frame_id % (uint32_t)every == 0) {
                char p[700];
                snprintf(p, sizeof p, "%s/frames/frame_%05u.ppm", out_dir, out_frame++);
                write_ppm(p, &fb);
            }
        }

        ps_alert_t al;
        while (ps_system_next_alert(&sys, &al)) {
            alerts_spoken++;
            if (!quiet) printf("  [%6.1fs] AUDIO: %s\n", ts, phrase_line(&al, line, sizeof line));
        }

        /* ---- navigation error bookkeeping ---- */
        float ex, ey;
        sim_world_estimate(w, &ex, &ey);
        float perr = hypotf(ex - w->tx, ey - w->ty);
        if (perr > max_pos_err) max_pos_err = perr;
        if (sys.nav.confidence < min_conf) min_conf = sys.nav.confidence;
        if (log && t % 200 == 0) {
            ps_nav_guidance_t g;
            ps_nav_guidance(&sys.nav, &g);
            fprintf(log, "%.2f,%d,%.3f,%.3f,%.1f,%.3f,%.3f,%.1f,%.3f,%.3f,%.3f,%d,%.1f,%.2f,%d,%d\n",
                    ts, w->phase, w->tx - 0.3f, w->ty, w->tyaw / DEG, ex - 0.3f, ey, sys.nav.yaw / DEG, perr, g.pos_sigma_m,
                    g.confidence, sys.alerts.level, g.route_bearing_rel_deg, g.route_dist_m, sys.nav.n_crumbs, sys.dets.n);
        }
    }
    if (log) fclose(log);
    if (meta) fclose(meta);
    const float tx = w->tx, ty = w->ty;
    const uint32_t t = w->t, frame_id = w->frame_id, t_out_start = w->t_out_start;
    const bool exited = w->exited, hose_used = w->hose_used;
    sim_scene_t scene = w->scene;

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
        o += snprintf(summary + o, sizeof summary - o, "detector %-7s    precision %.2f recall %.2f (tp %d fp %d fn %d) [%s detector]\n",
                      names[c], tp + fp ? (double)tp / (tp + fp) : 0, tp + fn ? (double)tp / (tp + fn) : 0, tp, fp, fn, model ? "AI model" : "threshold");
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
    o += snprintf(summary + o, sizeof summary - o, "host timing (us)    scene+pipeline mean %.0f p95 %.0f | detector mean %.0f p95 %.0f | render mean %.0f p95 %.0f\n",
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
