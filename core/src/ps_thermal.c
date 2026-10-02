#include "pyrosight/ps_thermal.h"

#include <string.h>

#define HIST_MIN_C (-40)
#define HIST_MAX_C (660)
#define HIST_BINS (HIST_MAX_C - HIST_MIN_C + 1)

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

void ps_thermal_init(ps_thermal_t *t)
{
    memset(t, 0, sizeof(*t));
}

/* Branch-light median of 9 (Paeth / Devillard network). */
#define SORT2(a, b) do { if ((a) > (b)) { ps_dc_t _t = (a); (a) = (b); (b) = _t; } } while (0)
static ps_dc_t median9(ps_dc_t p0, ps_dc_t p1, ps_dc_t p2, ps_dc_t p3, ps_dc_t p4,
                       ps_dc_t p5, ps_dc_t p6, ps_dc_t p7, ps_dc_t p8)
{
    SORT2(p1, p2); SORT2(p4, p5); SORT2(p7, p8); SORT2(p0, p1);
    SORT2(p3, p4); SORT2(p6, p7); SORT2(p1, p2); SORT2(p4, p5);
    SORT2(p7, p8); SORT2(p0, p3); SORT2(p5, p8); SORT2(p4, p7);
    SORT2(p3, p6); SORT2(p1, p4); SORT2(p2, p5); SORT2(p4, p7);
    SORT2(p4, p2); SORT2(p6, p4); SORT2(p4, p2);
    return p4;
}

void ps_thermal_denoise(ps_thermal_t *t, const ps_config_t *cfg,
                        ps_thermal_frame_t *f, ps_dc_t *scratch)
{
    /* 1. Temporal: motion-adaptive IIR in Q4. Pixels that change by more than
     *    the motion threshold snap to the new value, so moving people and
     *    flickering flames do not smear, while static sensor noise averages. */
    if (cfg->temporal_shift > 0) {
        const int32_t thr = (int32_t)cfg->motion_threshold_dc << 4;
        for (int i = 0; i < PS_THERM_PIXELS; i++) {
            int32_t in = (int32_t)f->px[i] << 4;
            if (!t->primed) {
                t->acc[i] = in;
            } else {
                int32_t d = in - t->acc[i];
                if (d > thr || d < -thr) t->acc[i] = in;
                else t->acc[i] += d >> cfg->temporal_shift;
            }
            f->px[i] = (ps_dc_t)((t->acc[i] + 8) >> 4);
        }
        t->primed = true;
    }

    /* 2. Spatial: 3x3 median removes dead/hot pixels and shot noise while
     *    keeping edges (a box blur would erase small distant people). */
    if (cfg->median3x3) {
        memcpy(scratch, f->px, sizeof(f->px));
        for (int y = 1; y < PS_THERM_H - 1; y++) {
            const ps_dc_t *r0 = &scratch[(y - 1) * PS_THERM_W];
            const ps_dc_t *r1 = &scratch[y * PS_THERM_W];
            const ps_dc_t *r2 = &scratch[(y + 1) * PS_THERM_W];
            for (int x = 1; x < PS_THERM_W - 1; x++) {
                f->px[y * PS_THERM_W + x] = median9(r0[x - 1], r0[x], r0[x + 1],
                                                    r1[x - 1], r1[x], r1[x + 1],
                                                    r2[x - 1], r2[x], r2[x + 1]);
            }
        }
    }

    ps_dc_t mn = f->px[0], mx = f->px[0];
    for (int i = 1; i < PS_THERM_PIXELS; i++) {
        if (f->px[i] < mn) mn = f->px[i];
        if (f->px[i] > mx) mx = f->px[i];
    }
    t->min_dc = mn;
    t->max_dc = mx;
}

/* Percentile over pixels below `exclude_from` (fire pixels are drawn with the
 * fire overlay, so they should not steal contrast from the rest of the room). */
static ps_dc_t percentile_below(const ps_thermal_frame_t *f, uint8_t pct, ps_dc_t exclude_from)
{
    uint16_t hist[HIST_BINS];
    memset(hist, 0, sizeof(hist));
    uint32_t n = 0;
    for (int i = 0; i < PS_THERM_PIXELS; i++) {
        if (f->px[i] >= exclude_from) continue;
        int c = f->px[i] / 10;
        hist[clampi(c, HIST_MIN_C, HIST_MAX_C) - HIST_MIN_C]++;
        n++;
    }
    if (n == 0) return exclude_from;
    uint32_t target = n * pct / 100u;
    uint32_t cum = 0;
    for (int b = 0; b < HIST_BINS; b++) {
        cum += hist[b];
        if (cum > target) return (ps_dc_t)((b + HIST_MIN_C) * 10);
    }
    return (ps_dc_t)(HIST_MAX_C * 10);
}

ps_dc_t ps_thermal_percentile(const ps_thermal_frame_t *f, uint8_t pct)
{
    return percentile_below(f, pct, INT16_MAX);
}

void ps_thermal_to_display(ps_thermal_t *t, const ps_config_t *cfg,
                           const ps_thermal_frame_t *f, ps_gray_frame_t *out)
{
    /* Window from scene percentiles, ignoring fire-hot pixels so a fire in
     * view does not crush everything else to black: those pixels saturate to
     * white (and get the fire overlay), and the top is capped as well. */
    int lo = percentile_below(f, cfg->lo_percentile, cfg->fire_min_dc);
    int hi = percentile_below(f, cfg->hi_percentile, cfg->fire_min_dc) + 10;
    if (hi > PS_DC(120)) hi = PS_DC(120);
    if (hi - lo < cfg->min_display_span_dc) {
        int mid = (hi + lo) / 2;
        lo = mid - cfg->min_display_span_dc / 2;
        hi = mid + cfg->min_display_span_dc / 2;
    }
    if (!t->win_primed) {
        t->win_lo_dc = (ps_dc_t)lo;
        t->win_hi_dc = (ps_dc_t)hi;
        t->win_primed = true;
    } else {
        /* Smooth the window to stop brightness pumping between frames. */
        t->win_lo_dc = (ps_dc_t)(t->win_lo_dc + (lo - t->win_lo_dc) / 4);
        t->win_hi_dc = (ps_dc_t)(t->win_hi_dc + (hi - t->win_hi_dc) / 4);
    }
    lo = t->win_lo_dc;
    hi = t->win_hi_dc;
    const int span = hi - lo > 1 ? hi - lo : 1;

    for (int i = 0; i < PS_THERM_PIXELS; i++)
        t->base[i] = (uint8_t)clampi((f->px[i] - lo) * 255 / span, 0, 255);

    if (cfg->detail_gain_q4 == 0) {
        memcpy(out->px, t->base, sizeof(out->px));
        return;
    }

    /* Separable 5x5 box blur (edge-clamped), then unsharp mask. */
    for (int y = 0; y < PS_THERM_H; y++) {
        const uint8_t *row = &t->base[y * PS_THERM_W];
        for (int x = 0; x < PS_THERM_W; x++) {
            int s = 0;
            for (int k = -2; k <= 2; k++) s += row[clampi(x + k, 0, PS_THERM_W - 1)];
            t->hsum[y * PS_THERM_W + x] = (uint16_t)s;
        }
    }
    for (int y = 0; y < PS_THERM_H; y++) {
        for (int x = 0; x < PS_THERM_W; x++) {
            int s = 0;
            for (int k = -2; k <= 2; k++)
                s += t->hsum[clampi(y + k, 0, PS_THERM_H - 1) * PS_THERM_W + x];
            int blur = s / 25;
            int b = t->base[y * PS_THERM_W + x];
            int v = b + (((b - blur) * cfg->detail_gain_q4) >> 4);
            out->px[y * PS_THERM_W + x] = (uint8_t)clampi(v, 0, 255);
        }
    }
}

uint8_t ps_thermal_model_code(ps_dc_t dc)
{
    int v = dc;
    if (v < PS_DC(-20)) return 0;
    if (v < PS_DC(60)) return (uint8_t)((v - PS_DC(-20)) * 191 / PS_DC(80));
    if (v < PS_DC(600)) return (uint8_t)(192 + (v - PS_DC(60)) * 63 / PS_DC(540));
    return 255;
}

void ps_thermal_to_model_input(const ps_thermal_frame_t *f, ps_gray_frame_t *out)
{
    for (int i = 0; i < PS_THERM_PIXELS; i++) out->px[i] = ps_thermal_model_code(f->px[i]);
}
