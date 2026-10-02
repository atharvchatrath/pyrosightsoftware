#include "scene.h"

#include <math.h>
#include <string.h>

#define PI_F 3.14159265f

static unsigned rng = 12345u;
static float fpn[PS_THERM_W]; /* column fixed-pattern noise, deci-C */

float sim_rand01(void)
{
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (rng & 0xFFFFFF) / 16777216.0f;
}

float sim_randn(void)
{
    float u1 = sim_rand01() + 1e-7f, u2 = sim_rand01();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI_F * u2);
}

void sim_scene_seed(unsigned seed)
{
    rng = seed ? seed : 1u;
    for (int x = 0; x < PS_THERM_W; x++) fpn[x] = 2.0f * sim_randn();
}

static float smoothstep(float e0, float e1, float v)
{
    float t = (v - e0) / (e1 - e0);
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    return t * t * (3 - 2 * t);
}

/* Air/surface temperature by height: cool near the floor, hot gas layer above. */
static float temp_at_height(const sim_scene_t *sc, float z)
{
    return sc->ambient_c + (sc->layer_c - sc->ambient_c) * smoothstep(0.9f, sc->wall_height, z);
}

/* Ray vs segment: distance along the ray, or -1. */
static float ray_seg(float ox, float oy, float dx, float dy, const sim_wall_t *w)
{
    float ex = w->x1 - w->x0, ey = w->y1 - w->y0;
    float den = dx * ey - dy * ex;
    if (fabsf(den) < 1e-9f) return -1;
    float t = ((w->x0 - ox) * ey - (w->y0 - oy) * ex) / den;
    float u = ((w->x0 - ox) * dy - (w->y0 - oy) * dx) / den;
    if (t <= 1e-4f || u < 0 || u > 1) return -1;
    return t;
}

bool sim_collide(const sim_scene_t *sc, float ax, float ay, float *bx, float *by)
{
    float dx = *bx - ax, dy = *by - ay;
    float len = sqrtf(dx * dx + dy * dy);
    if (len < 1e-6f) return false;
    const float margin = 0.25f; /* body radius */
    bool hit = false;
    for (int i = 0; i < sc->n_walls; i++) {
        float t = ray_seg(ax, ay, dx / len, dy / len, &sc->walls[i]);
        if (t >= 0 && t < len + margin) {
            /* Axis-aligned walls: slide along them. */
            if (sc->walls[i].x0 == sc->walls[i].x1) *bx = ax; else *by = ay;
            dx = *bx - ax; dy = *by - ay;
            len = sqrtf(dx * dx + dy * dy);
            hit = true;
            if (len < 1e-6f) return true;
        }
    }
    return hit;
}

/* Smoke: LWIR passes smoke far better than visible light, but dense smoke still
 * pulls apparent temperatures toward the smoke's own temperature. */
static float through_smoke(const sim_scene_t *sc, float t_c, float dist)
{
    float k = 0.12f * sc->smoke;      /* 1/m extinction in the 8-14 um band */
    float tau = expf(-k * dist);
    float smoke_c = sc->ambient_c + 0.3f * (sc->layer_c - sc->ambient_c);
    return t_c * tau + smoke_c * (1 - tau);
}

void sim_render(const sim_scene_t *sc, float px, float py, float yaw, float cam_h,
                float t_s, ps_thermal_frame_t *out, sim_truth_t *truth)
{
    const float f = (PS_THERM_W * 0.5f) / tanf(57.0f * 0.5f * PI_F / 180.0f);
    static float img[PS_THERM_PIXELS];
    static float depth[PS_THERM_W];
    static int16_t owner[PS_THERM_PIXELS]; /* which object drew the pixel */

    /* 1. Walls, ceiling and floor by ray casting each column. */
    for (int x = 0; x < PS_THERM_W; x++) {
        float l = (PS_THERM_W * 0.5f - x - 0.5f) / f; /* + = left */
        float a = atanf(l);
        float dx = cosf(yaw + a), dy = sinf(yaw + a);
        float best = 1e9f;
        int wi = -1;
        for (int i = 0; i < sc->n_walls; i++) {
            float t = ray_seg(px, py, dx, dy, &sc->walls[i]);
            if (t > 0 && t < best) { best = t; wi = i; }
        }
        float dp = wi >= 0 ? best * cosf(a) : 40.0f;
        depth[x] = dp;
        /* Position along the wall gives texture: studs every 0.6 m, door frames. */
        float hx = px + dx * best, hy = py + dy * best;
        float along = wi >= 0 ? (sc->walls[wi].x0 == sc->walls[wi].x1 ? hy : hx) : 0;
        float stud = (fmodf(fabsf(along), 0.6f) < 0.05f) ? -0.8f : 0.0f;
        float wall_off = wi >= 0 ? (float)((wi * 7) % 5) * 0.6f - 1.2f : 0;

        for (int y = 0; y < PS_THERM_H; y++) {
            float v = (PS_THERM_H * 0.5f - y - 0.5f) / f; /* + = up */
            float z = cam_h + v * dp;
            float tc, dist;
            if (wi < 0 && z > 0) { tc = sc->outside_c; dist = 40; }
            else if (z >= 0 && z <= sc->wall_height) {
                tc = temp_at_height(sc, z) + wall_off + stud;
                dist = dp;
            } else if (z > sc->wall_height) {
                float dc = (sc->wall_height - cam_h) / v;
                tc = sc->layer_c + 4.0f * sinf(dc * 1.3f + t_s * 0.7f); /* rolling gas */
                dist = dc;
            } else {
                float dfl = cam_h / -v;
                tc = sc->ambient_c - 2.0f + 0.5f * sinf(dfl * 2.0f);
                dist = dfl;
            }
            img[y * PS_THERM_W + x] = through_smoke(sc, tc, dist);
            owner[y * PS_THERM_W + x] = -1;
        }
    }

    /* 2. Objects as billboards, far to near, depth-tested against walls. */
    int order[32];
    float odepth[32];
    int n = sc->n_objs < 32 ? sc->n_objs : 32;
    const float cy = cosf(yaw), sy = sinf(yaw);
    for (int i = 0; i < n; i++) {
        float rx = sc->objs[i].x - px, ry = sc->objs[i].y - py;
        odepth[i] = rx * cy + ry * sy;
        order[i] = i;
    }
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && odepth[order[j]] > odepth[order[j - 1]]; j--) {
            int t = order[j]; order[j] = order[j - 1]; order[j - 1] = t;
        }

    for (int oi = 0; oi < n; oi++) {
        const int idx = order[oi];
        const sim_obj_t *o = &sc->objs[idx];
        float rx = o->x - px, ry = o->y - py;
        float fwd = rx * cy + ry * sy, left = -rx * sy + ry * cy;
        if (fwd < 0.4f) continue;
        float w = o->width;
        if (o->kind == OBJ_PERSON_LYING) {
            /* Apparent width of a 1.7 m body lying at angle to the view ray. */
            float rel = o->yaw - atan2f(ry, rx);
            w = 1.7f * fabsf(sinf(rel)) + 0.45f * fabsf(cosf(rel));
        }
        float sx = PS_THERM_W * 0.5f - f * left / fwd;
        float hw = f * w * 0.5f / fwd;
        float ytop = PS_THERM_H * 0.5f - f * (o->z0 + o->height - cam_h) / fwd;
        float ybot = PS_THERM_H * 0.5f - f * (o->z0 - cam_h) / fwd;
        int x0 = (int)floorf(sx - hw), x1 = (int)ceilf(sx + hw);
        int y0 = (int)floorf(ytop), y1 = (int)ceilf(ybot);
        for (int y = y0; y < y1; y++) {
            if (y < 0 || y >= PS_THERM_H) continue;
            float vy = (y + 0.5f - ytop) / (ybot - ytop); /* 0 top .. 1 bottom */
            for (int x = x0; x < x1; x++) {
                if (x < 0 || x >= PS_THERM_W || fwd > depth[x]) continue;
                float ux = (x + 0.5f - (sx - hw)) / (2 * hw) - 0.5f; /* -0.5..0.5 */
                float tc;
                bool inside = false;
                switch (o->kind) {
                case OBJ_PERSON_STANDING: {
                    /* Head (top 13%), shoulders/torso, legs split. */
                    if (vy < 0.13f) { inside = ux * ux * 4.0f + (vy - 0.065f) * (vy - 0.065f) * 300.0f < 0.6f; tc = 35.0f; }
                    else if (vy < 0.55f) { inside = fabsf(ux) < 0.5f - 0.1f * (vy - 0.13f); tc = 31.0f + 1.5f * (1 - fabsf(ux) * 2); }
                    else { inside = fabsf(ux) < 0.42f && fabsf(ux) > 0.04f; tc = 30.0f; }
                    break;
                }
                case OBJ_PERSON_LYING: {
                    float e = ux * ux * 4.0f + (vy - 0.5f) * (vy - 0.5f) * 4.0f;
                    inside = e < 1.0f;
                    tc = 31.0f + 3.0f * (1 - e);
                    break;
                }
                case OBJ_FIRE: {
                    /* Flame: narrows with height, flickers. */
                    float flick = 0.15f * sinf(t_s * 9.0f + ux * 11.0f) + 0.1f * sinf(t_s * 15.0f + vy * 7.0f);
                    float half = 0.5f * (0.35f + 0.65f * vy) + flick * (1 - vy);
                    inside = fabsf(ux) < half;
                    tc = 280.0f + 300.0f * vy * (1 - 2 * fabsf(ux)) + 60.0f * flick;
                    break;
                }
                case OBJ_HOT_PIPE:
                default:
                    inside = true;
                    tc = 85.0f;
                    break;
                }
                if (!inside) continue;
                img[y * PS_THERM_W + x] = through_smoke(sc, tc, fwd);
                owner[y * PS_THERM_W + x] = (int16_t)idx;
            }
        }
    }

    /* 3. Ground truth: tight boxes over the visible pixels of each object. */
    if (truth) {
        truth->n = 0;
        for (int i = 0; i < n && truth->n < PS_MAX_DETECTIONS; i++) {
            const sim_obj_t *o = &sc->objs[i];
            if (o->kind == OBJ_HOT_PIPE) continue;
            int bx0 = PS_THERM_W, by0 = PS_THERM_H, bx1 = -1, by1 = -1, cnt = 0;
            for (int p = 0; p < PS_THERM_PIXELS; p++) {
                if (owner[p] != i) continue;
                int x = p % PS_THERM_W, y = p / PS_THERM_W;
                if (x < bx0) bx0 = x;
                if (x > bx1) bx1 = x;
                if (y < by0) by0 = y;
                if (y > by1) by1 = y;
                cnt++;
            }
            if (cnt < 6) continue;
            ps_detection_t *g = &truth->gt[truth->n++];
            memset(g, 0, sizeof(*g));
            g->cls = o->kind == OBJ_FIRE ? PS_CLASS_FIRE : PS_CLASS_PERSON;
            g->x = (float)bx0;
            g->y = (float)by0;
            g->w = (float)(bx1 - bx0 + 1);
            g->h = (float)(by1 - by0 + 1);
            g->score = 1;
            g->dist_m = (o->x - px) * cy + (o->y - py) * sy;
        }
    }

    /* 4. Sensor: temporal noise, column fixed-pattern noise, quantisation. */
    for (int i = 0; i < PS_THERM_PIXELS; i++) {
        float v = img[i] * 10.0f + sc->noise_c * 10.0f * sim_randn() + fpn[i % PS_THERM_W];
        if (v > 32000) v = 32000;
        if (v < -32000) v = -32000;
        out->px[i] = (ps_dc_t)lrintf(v);
    }
}
