/*
 * Browser build of the simulator: the same core + sim_world + reference NN
 * runtime, compiled to WebAssembly (then optionally to plain JavaScript with
 * binaryen's wasm2js). The page (sim/web/index.html) drives it one thermal
 * frame at a time and reads results straight out of linear memory.
 *
 * All buffers are static; every export is a plain C function.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../nn_ref.h"
#include "../sim_world.h"

#define API __attribute__((visibility("default")))
#define GRID_W 40
#define GRID_H 30

enum { DET_NEURAL = 0, DET_CLASSICAL = 1, DET_TRUTH = 2 };

static ps_system_t sys;
static sim_world_t world;
static ps_fb_t fb;
static uint8_t rgba[PS_DISP_W * PS_DISP_H * 4];
static float state[48];
static float crumbs[PS_NAV_MAX_CRUMBS * 2];
static float boxes[PS_MAX_DETECTIONS * 8];
static float geom[256];
static char alert_text[256];
static int detector = DET_NEURAL;
static bool ready;

static uint8_t *model_blob;
static nn_ref_t *model;
static float heat[2 * GRID_W * GRID_H], whs[2 * GRID_W * GRID_H], offs[2 * GRID_W * GRID_H];

API uint8_t *api_model_alloc(int len)
{
    nn_ref_free(model);
    model = NULL;
    free(model_blob);
    model_blob = malloc((size_t)len);
    return model_blob;
}

API int api_model_load(int len)
{
    model = model_blob ? nn_ref_load(model_blob, (size_t)len) : NULL;
    return model != NULL;
}

API int api_n_scenarios(void) { return SIM_N_SCENARIOS; }
API const char *api_scenario_name(int i) { return i >= 0 && i < SIM_N_SCENARIOS ? SIM_SCENARIOS[i].name : ""; }

/* smoke < 0 keeps the scenario's own value. */
API int api_init(int scenario, unsigned seed, float smoke, int heading_snap)
{
    if (scenario < 0 || scenario >= SIM_N_SCENARIOS) return 0;
    ready = sim_world_init(&world, &sys, SIM_SCENARIOS[scenario].name, seed, smoke, -99.0f, heading_snap != 0);
    alert_text[0] = 0;
    if (ready) ps_system_render(&sys, &fb, 0);
    return ready;
}

API void api_set_detector(int d) { detector = d; }
API void api_set_smoke(float s) { world.scene.smoke = s < 0 ? 0 : (s > 1 ? 1 : s); }
API float api_get_smoke(void) { return world.scene.smoke; }
API void api_button(int b) { if (ready) ps_system_on_button(&sys, (ps_button_t)b, world.t); }

static void run_detector(void)
{
    ps_detections_t d;
    memset(&d, 0, sizeof d);
    if (detector == DET_NEURAL && model) {
        if (nn_ref_run(model, sys.model_input.px, heat, whs, offs) == 0) {
            ps_centernet_decode(heat, whs, offs, GRID_W, GRID_H, 4, sys.cfg.score_threshold, &d);
            ps_nms(&d, sys.cfg.nms_iou);
        }
        d.source = PS_DETECTOR_NEURAL;
        d.frame_id = sys.frame.frame_id;
        d.t_ms = sys.frame.t_ms;
        ps_system_on_detections(&sys, &d, world.t);
    } else if (detector == DET_TRUTH) {
        sim_world_truth_detections(&world, &d);
        ps_system_on_detections(&sys, &d, world.t);
    } else {
        ps_system_run_classical(&sys);
    }
}

/* Advance until the next thermal frame (115 ms of simulated time), run the
 * chosen detector on it, composite the eyepiece. Returns 1 for a frame, 2 when
 * the run is over, 0 if not initialised. */
API int api_run_frame(void)
{
    if (!ready) return 0;
    const uint32_t f0 = world.frame_id;
    int ev = SIM_EV_NONE;
    while (world.frame_id == f0) {
        ev = sim_world_step(&world);
        if (ev == SIM_EV_DONE) break;
        if (ev == SIM_EV_FRAME) run_detector();
    }
    ps_system_render(&sys, &fb, world.t);
    return ev == SIM_EV_DONE ? 2 : 1;
}

/* Re-composite the current frame without advancing (after a palette change). */
API void api_render(void) { if (ready) ps_system_render(&sys, &fb, world.t); }

API uint8_t *api_rgba(void)
{
    for (int i = 0; i < PS_DISP_W * PS_DISP_H; i++) {
        const uint16_t c = fb.px[i];
        uint8_t r = (uint8_t)((c >> 8) & 0xF8), g = (uint8_t)((c >> 3) & 0xFC), b = (uint8_t)((c << 3) & 0xF8);
        rgba[4 * i] = r | r >> 5;
        rgba[4 * i + 1] = g | g >> 6;
        rgba[4 * i + 2] = b | b >> 5;
        rgba[4 * i + 3] = 255;
    }
    return rgba;
}

/* Layout documented in index.html (STATE_*). */
API float *api_state(void)
{
    ps_nav_guidance_t g;
    ps_nav_guidance(&sys.nav, &g);
    float ex, ey;
    sim_world_estimate(&world, &ex, &ey);
    int np = 0, nf = 0;
    for (int i = 0; i < sys.dets.n; i++) { if (sys.dets.d[i].cls == PS_CLASS_PERSON) np++; else nf++; }
    float *s = state;
    s[0] = world.t / 1000.0f;
    s[1] = world.tx; s[2] = world.ty; s[3] = world.tyaw;
    s[4] = ex; s[5] = ey; s[6] = sys.nav.yaw;
    s[7] = (float)world.phase;
    s[8] = sys.nav.confidence;
    s[9] = (float)sys.alerts.level;
    s[10] = sys.nav.pos_sigma_m;
    s[11] = world.hose_used; s[12] = world.exited; s[13] = world.done;
    s[14] = sys.camera_ok; s[15] = sys.imu_ok || !sys.imu_seen;
    s[16] = (float)sys.nav.n_crumbs; s[17] = (float)sys.nav.return_target;
    s[18] = g.valid; s[19] = g.route_dist_m; s[20] = g.home_dist_m; s[21] = g.route_bearing_rel_deg;
    s[22] = (float)np; s[23] = (float)nf;
    s[24] = (float)world.frame_id;
    s[25] = sys.nav.dist_walked_m;
    s[26] = (float)sys.nav.steps;
    s[27] = hypotf(ex - world.tx, ey - world.ty);
    s[28] = g.exit_is_next;
    s[29] = (float)sys.dets.source;
    s[30] = world.scene.smoke;
    s[31] = (float)sys.disp.palette;
    s[32] = (float)world.truth.n;
    return state;
}

/* Breadcrumbs in world coordinates, x,y pairs; count in state[16]. */
API float *api_crumbs(void)
{
    for (int i = 0; i < sys.nav.n_crumbs; i++) {
        crumbs[2 * i] = sys.nav.crumbs[i].p.x + 0.3f; /* entry frame -> world (START_X) */
        crumbs[2 * i + 1] = sys.nav.crumbs[i].p.y;
    }
    return crumbs;
}

/* Current detections: cls, score, x, y, w, h, dist_m, 0 (count in return). */
API float *api_boxes(void) { return boxes; }
API int api_n_boxes(void)
{
    for (int i = 0; i < sys.dets.n; i++) {
        const ps_detection_t *d = &sys.dets.d[i];
        float *b = &boxes[8 * i];
        b[0] = (float)d->cls; b[1] = d->score; b[2] = d->x; b[3] = d->y; b[4] = d->w; b[5] = d->h; b[6] = d->dist_m; b[7] = 0;
    }
    return sys.dets.n;
}

/* Static geometry for the map: walls (x0,y0,x1,y1)... then objects (kind,x,y,width). */
API int api_n_walls(void) { return SIM_N_WALLS; }
API int api_n_objs(void) { return SIM_N_OBJS; }
API float *api_geometry(void)
{
    int k = 0;
    for (int i = 0; i < SIM_N_WALLS && k + 4 <= (int)(sizeof geom / sizeof *geom); i++) {
        geom[k++] = SIM_WALLS[i].x0; geom[k++] = SIM_WALLS[i].y0;
        geom[k++] = SIM_WALLS[i].x1; geom[k++] = SIM_WALLS[i].y1;
    }
    for (int i = 0; i < SIM_N_OBJS && k + 4 <= (int)(sizeof geom / sizeof *geom); i++) {
        geom[k++] = (float)SIM_OBJS[i].kind; geom[k++] = SIM_OBJS[i].x;
        geom[k++] = SIM_OBJS[i].y; geom[k++] = SIM_OBJS[i].width;
    }
    return geom;
}

/* The scenario's inbound route (the hose line), x,y pairs. */
API int api_n_inbound(void) { return ready ? world.sc->n_inbound : 0; }
API float *api_inbound(void)
{
    static float p[32];
    for (int i = 0; ready && i < world.sc->n_inbound; i++) { p[2 * i] = world.sc->inbound[i].x; p[2 * i + 1] = world.sc->inbound[i].y; }
    return p;
}

/* Next spoken alert as text, or "" when the queue is empty. */
API const char *api_next_alert(void)
{
    ps_alert_t a;
    alert_text[0] = 0;
    if (!ready || !ps_system_next_alert(&sys, &a)) return alert_text;
    size_t n = 0;
    for (int i = 0; i < a.n_parts; i++) {
        const char *s = ps_phrase_text[a.parts[i]];
        size_t l = strlen(s);
        if (n + l + 2 >= sizeof alert_text) break;
        if (n) alert_text[n++] = ' ';
        memcpy(alert_text + n, s, l);
        n += l;
        alert_text[n] = 0;
    }
    return alert_text;
}

/* Benchmark hook: run the model once on the current model input. */
API int api_infer_once(void)
{
    if (!model) return -1;
    return nn_ref_run(model, sys.model_input.px, heat, whs, offs);
}
