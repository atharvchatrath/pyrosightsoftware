#include "pyrosight/ps_display.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

extern const uint8_t ps_font5x7[59][5];

#define BLACK 0x0000u

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

uint16_t ps_rgb565(uint8_t r, uint8_t g, uint8_t b)
{
    return (uint16_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
}

void ps_display_settings_default(ps_display_settings_t *s)
{
    s->palette = PS_PALETTE_WHITE_HOT;
    s->brightness = 160;
    s->fire_overlay = true;
    s->show_home_bearing = true;
}

/* ---------------------------------------------------------------- palettes */

static uint16_t lut[PS_PALETTE_COUNT][256];
static bool lut_ready;

static void build_luts(void)
{
    static const uint8_t iron[5][3] = {
        {0, 0, 0}, {80, 0, 140}, {200, 30, 60}, {250, 140, 0}, {255, 255, 220},
    };
    for (int v = 0; v < 256; v++) {
        lut[PS_PALETTE_WHITE_HOT][v] = ps_rgb565((uint8_t)v, (uint8_t)v, (uint8_t)v);
        float t = v / 255.0f * 4.0f;
        int k = clampi((int)t, 0, 3);
        float f = t - k;
        uint8_t rgb[3];
        for (int c = 0; c < 3; c++) rgb[c] = (uint8_t)(iron[k][c] + (iron[k + 1][c] - iron[k][c]) * f);
        lut[PS_PALETTE_IRON][v] = ps_rgb565(rgb[0], rgb[1], rgb[2]);
        /* Amber at ~80% luminance: easier on dark-adapted eyes than white. */
        int a = v * 205 / 255;
        lut[PS_PALETTE_NIGHT][v] = ps_rgb565((uint8_t)a, (uint8_t)(a * 140 / 255), 0);
    }
    lut_ready = true;
}

/* -------------------------------------------------------------- primitives */

static inline void put(ps_fb_t *fb, int x, int y, uint16_t c)
{
    if ((unsigned)x < PS_DISP_W && (unsigned)y < PS_DISP_H) fb->px[y * PS_DISP_W + x] = c;
}

void ps_fb_fill(ps_fb_t *fb, uint16_t c)
{
    for (int i = 0; i < PS_DISP_W * PS_DISP_H; i++) fb->px[i] = c;
}

void ps_fb_fill_rect(ps_fb_t *fb, int x, int y, int w, int h, uint16_t c)
{
    int x0 = clampi(x, 0, PS_DISP_W), x1 = clampi(x + w, 0, PS_DISP_W);
    int y0 = clampi(y, 0, PS_DISP_H), y1 = clampi(y + h, 0, PS_DISP_H);
    for (int yy = y0; yy < y1; yy++)
        for (int xx = x0; xx < x1; xx++) fb->px[yy * PS_DISP_W + xx] = c;
}

void ps_fb_rect(ps_fb_t *fb, int x, int y, int w, int h, int t, uint16_t c)
{
    ps_fb_fill_rect(fb, x, y, w, t, c);
    ps_fb_fill_rect(fb, x, y + h - t, w, t, c);
    ps_fb_fill_rect(fb, x, y, t, h, c);
    ps_fb_fill_rect(fb, x + w - t, y, t, h, c);
}

void ps_fb_line(ps_fb_t *fb, int x0, int y0, int x1, int y1, uint16_t c)
{
    int dx = x1 > x0 ? x1 - x0 : x0 - x1, sx = x0 < x1 ? 1 : -1;
    int dy = y1 > y0 ? y0 - y1 : y1 - y0, sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        put(fb, x0, y0, c);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void ps_fb_triangle(ps_fb_t *fb, int x0, int y0, int x1, int y1, int x2, int y2, uint16_t c)
{
    int minx = clampi(fminf(x0, fminf(x1, x2)), 0, PS_DISP_W - 1);
    int maxx = clampi(fmaxf(x0, fmaxf(x1, x2)), 0, PS_DISP_W - 1);
    int miny = clampi(fminf(y0, fminf(y1, y2)), 0, PS_DISP_H - 1);
    int maxy = clampi(fmaxf(y0, fmaxf(y1, y2)), 0, PS_DISP_H - 1);
    int area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if (area == 0) return;
    for (int y = miny; y <= maxy; y++)
        for (int x = minx; x <= maxx; x++) {
            int w0 = (x1 - x) * (y2 - y) - (x2 - x) * (y1 - y);
            int w1 = (x2 - x) * (y0 - y) - (x0 - x) * (y2 - y);
            int w2 = (x0 - x) * (y1 - y) - (x1 - x) * (y0 - y);
            if ((area > 0 && w0 >= 0 && w1 >= 0 && w2 >= 0) ||
                (area < 0 && w0 <= 0 && w1 <= 0 && w2 <= 0))
                fb->px[y * PS_DISP_W + x] = c;
        }
}

void ps_fb_circle(ps_fb_t *fb, int cx, int cy, int r, uint16_t c)
{
    int x = r, y = 0, err = 1 - r;
    while (x >= y) {
        put(fb, cx + x, cy + y, c); put(fb, cx + y, cy + x, c);
        put(fb, cx - y, cy + x, c); put(fb, cx - x, cy + y, c);
        put(fb, cx - x, cy - y, c); put(fb, cx - y, cy - x, c);
        put(fb, cx + y, cy - x, c); put(fb, cx + x, cy - y, c);
        y++;
        if (err < 0) err += 2 * y + 1;
        else { x--; err += 2 * (y - x) + 1; }
    }
}

static void glyph(ps_fb_t *fb, int x, int y, char ch, int scale, uint16_t c)
{
    if (ch >= 'a' && ch <= 'z') ch = (char)(ch - 32);
    if (ch < 0x20 || ch > 0x5A) ch = '?';
    const uint8_t *g = ps_font5x7[ch - 0x20];
    for (int col = 0; col < 5; col++)
        for (int row = 0; row < 7; row++)
            if (g[col] & (1u << row)) ps_fb_fill_rect(fb, x + col * scale, y + row * scale, scale, scale, c);
}

int ps_fb_text_width(const char *s, int scale)
{
    int n = (int)strlen(s);
    return n ? n * 6 * scale - scale : 0;
}

int ps_fb_text(ps_fb_t *fb, int x, int y, const char *s, int scale, uint16_t c, bool outline)
{
    if (outline) {
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                if (!dx && !dy) continue;
                int xx = x + dx * scale / 2 + dx;
                for (const char *p = s; *p; p++, xx += 6 * scale) glyph(fb, xx, y + dy * scale / 2 + dy, *p, scale, BLACK);
            }
    }
    int xx = x;
    for (const char *p = s; *p; p++, xx += 6 * scale) glyph(fb, xx, y, *p, scale, c);
    return xx - x;
}

static void thick_rect(ps_fb_t *fb, int x, int y, int w, int h, uint16_t c)
{
    ps_fb_rect(fb, x - 1, y - 1, w + 2, h + 2, 4, BLACK); /* halo */
    ps_fb_rect(fb, x, y, w, h, 2, c);
}

/* Corner brackets for people: less clutter over the person's own pixels. */
static void brackets(ps_fb_t *fb, int x, int y, int w, int h, uint16_t c)
{
    int l = (w < h ? w : h) / 3;
    if (l < 4) l = 4;
    for (int pass = 0; pass < 2; pass++) {
        uint16_t col = pass ? c : BLACK;
        int t = pass ? 2 : 4, o = pass ? 0 : 1;
        ps_fb_fill_rect(fb, x - o, y - o, l + o, t, col);         ps_fb_fill_rect(fb, x - o, y - o, t, l + o, col);
        ps_fb_fill_rect(fb, x + w - l, y - o, l + o, t, col);     ps_fb_fill_rect(fb, x + w - t + o, y - o, t, l + o, col);
        ps_fb_fill_rect(fb, x - o, y + h - t + o, l + o, t, col); ps_fb_fill_rect(fb, x - o, y + h - l, t, l + o, col);
        ps_fb_fill_rect(fb, x + w - l, y + h - t + o, l + o, t, col);
        ps_fb_fill_rect(fb, x + w - t + o, y + h - l, t, l + o, col);
    }
}

/* ------------------------------------------------------------------- HUD */

static void draw_arrow(ps_fb_t *fb, int cx, int cy, int r, float rel_deg, uint16_t c)
{
    /* Up = straight ahead; positive bearing = to the left (counter-clockwise). */
    float a = rel_deg * 3.14159265f / 180.0f;
    float fx = -sinf(a), fy = -cosf(a); /* forward unit vector on screen */
    float px = -fy, py = fx;            /* perpendicular */
    int tipx = cx + (int)(fx * r), tipy = cy + (int)(fy * r);
    int bx = cx - (int)(fx * r * 0.55f), by = cy - (int)(fy * r * 0.55f);
    int lx = bx + (int)(px * r * 0.6f), ly = by + (int)(py * r * 0.6f);
    int rx = bx - (int)(px * r * 0.6f), ry = by - (int)(py * r * 0.6f);
    int nx = cx - (int)(fx * r * 0.2f), ny = cy - (int)(fy * r * 0.2f); /* notch */
    /* Black halo: same shape, slightly larger. */
    ps_fb_triangle(fb, tipx + (int)(fx * 2), tipy + (int)(fy * 2), lx + (int)(px * 2), ly + (int)(py * 2), nx, ny, BLACK);
    ps_fb_triangle(fb, tipx + (int)(fx * 2), tipy + (int)(fy * 2), nx, ny, rx - (int)(px * 2), ry - (int)(py * 2), BLACK);
    ps_fb_triangle(fb, tipx, tipy, lx, ly, nx, ny, c);
    ps_fb_triangle(fb, tipx, tipy, nx, ny, rx, ry, c);
}

static void draw_nav(ps_fb_t *fb, const ps_display_settings_t *s, const ps_hud_t *h)
{
    const int cx = PS_DISP_W / 2, cy = 30, r = 22;
    const uint16_t green = ps_rgb565(40, 255, 80), yellow = ps_rgb565(255, 220, 0);
    const uint16_t red = ps_rgb565(255, 40, 40), white = 0xFFFF;
    char buf[24];

    if (!h->nav_tracking) {
        ps_fb_text(fb, cx - ps_fb_text_width("MARK ENTRY", 1) / 2, 6, "MARK ENTRY", 1, white, true);
        return;
    }
    if (!h->nav || !h->nav->valid) return;
    const ps_nav_guidance_t *g = h->nav;

    if (h->nav_level == PS_NAVCONF_UNRELIABLE) {
        /* Do not show an arrow we do not believe: tell them to use the hose. */
        if ((h->t_ms / 500) % 2 == 0) {
            const char *m = "FOLLOW HOSE";
            ps_fb_text(fb, cx - ps_fb_text_width(m, 2) / 2, 10, m, 2, red, true);
        }
        snprintf(buf, sizeof buf, "NAV +-%dM", (int)(g->pos_sigma_m + 0.5f));
        ps_fb_text(fb, cx - ps_fb_text_width(buf, 1) / 2, 30, buf, 1, red, true);
        return;
    }

    uint16_t col = h->nav_level == PS_NAVCONF_GOOD ? green : yellow;
    ps_fb_circle(fb, cx, cy, r + 3, BLACK);
    ps_fb_circle(fb, cx, cy, r + 2, col);
    ps_fb_circle(fb, cx, cy, r + 1, BLACK);
    if (g->route_dist_m < 1.0f) {
        ps_fb_text(fb, cx - ps_fb_text_width("EXIT", 2) / 2, cy - 7, "EXIT", 2, col, true);
    } else {
        draw_arrow(fb, cx, cy, r - 2, g->route_bearing_rel_deg, col);
    }
    if (s->show_home_bearing) {
        /* Small dot on the ring: straight-line direction to the door. */
        float a = g->home_bearing_rel_deg * 3.14159265f / 180.0f;
        int dx = cx + (int)(-sinf(a) * (r + 2)), dy = cy + (int)(-cosf(a) * (r + 2));
        ps_fb_fill_rect(fb, dx - 3, dy - 3, 7, 7, BLACK);
        ps_fb_fill_rect(fb, dx - 2, dy - 2, 5, 5, white);
    }
    snprintf(buf, sizeof buf, "%dM", (int)(g->route_dist_m + 0.5f));
    ps_fb_text(fb, cx - ps_fb_text_width(buf, 1) / 2, cy + r + 6, buf, 1, col, true);
}

static void draw_status(ps_fb_t *fb, const ps_hud_t *h)
{
    const uint16_t white = 0xFFFF, red = ps_rgb565(255, 40, 40), grey = ps_rgb565(180, 180, 180);
    char buf[16];
    const int y = PS_DISP_H - 10;

    /* Battery, bottom-right. */
    snprintf(buf, sizeof buf, "%d%%", h->battery_pct);
    bool blink = (h->t_ms / 400) % 2 == 0;
    if (!h->battery_low || blink)
        ps_fb_text(fb, PS_DISP_W - 4 - ps_fb_text_width(buf, 1), y, buf, 1, h->battery_low ? red : grey, true);

    /* Detector source, bottom-left: AI = neural, TH = threshold fallback. */
    const char *src = h->detector == PS_DETECTOR_NEURAL ? "AI" : h->detector == PS_DETECTOR_CLASSICAL ? "TH" : "--";
    ps_fb_text(fb, 4, y, src, 1, grey, true);

    if (!h->imu_ok) ps_fb_text(fb, 24, y, "NO IMU", 1, red, true);
    (void)white;
}

void ps_display_render(ps_fb_t *fb, const ps_display_settings_t *s,
                       const ps_config_t *cfg, const ps_hud_t *h)
{
    if (!lut_ready) build_luts();
    const uint16_t *pal = lut[s->palette < PS_PALETTE_COUNT ? s->palette : 0];
    const uint16_t fire_col = ps_rgb565(255, 90, 0);

    /* 1. Thermal image, nearest-neighbour 2x (keeps edges crisp). */
    if (h->camera_ok && h->thermal) {
        for (int y = 0; y < PS_THERM_H; y++) {
            for (int x = 0; x < PS_THERM_W; x++) {
                int i = y * PS_THERM_W + x;
                uint16_t c = pal[h->thermal->px[i]];
                if (s->fire_overlay && h->radiometric && h->radiometric->px[i] >= cfg->fire_min_dc)
                    c = fire_col;
                uint16_t *d = &fb->px[(y * PS_DISP_SCALE) * PS_DISP_W + x * PS_DISP_SCALE];
                d[0] = d[1] = c;
                d[PS_DISP_W] = d[PS_DISP_W + 1] = c;
            }
        }
    } else {
        ps_fb_fill(fb, BLACK);
        if ((h->t_ms / 500) % 2 == 0) {
            const char *m = "THERMAL LOST";
            ps_fb_text(fb, (PS_DISP_W - ps_fb_text_width(m, 3)) / 2, PS_DISP_H / 2 - 10, m, 3,
                       ps_rgb565(255, 40, 40), true);
        }
    }

    /* 2. Detections. */
    if (h->camera_ok && h->dets) {
        char buf[16];
        for (int i = 0; i < h->dets->n; i++) {
            const ps_detection_t *d = &h->dets->d[i];
            int x = (int)(d->x * PS_DISP_SCALE), y = (int)(d->y * PS_DISP_SCALE);
            int w = (int)(d->w * PS_DISP_SCALE), hh = (int)(d->h * PS_DISP_SCALE);
            if (d->cls == PS_CLASS_FIRE) {
                thick_rect(fb, x, y, w, hh, ps_rgb565(255, 40, 0));
            } else {
                uint16_t c = ps_rgb565(0, 230, 255);
                brackets(fb, x, y, w, hh, c);
                if (d->dist_m > 0.0f) {
                    if (d->dist_m < 10.0f) snprintf(buf, sizeof buf, "%s%d.%dM", d->truncated ? "<" : "",
                                                    (int)d->dist_m, (int)(d->dist_m * 10) % 10);
                    else snprintf(buf, sizeof buf, "%s%dM", d->truncated ? "<" : "", (int)(d->dist_m + 0.5f));
                    int ty = y - 10 < 0 ? y + hh + 2 : y - 10;
                    ps_fb_text(fb, x, ty, buf, 1, c, true);
                }
            }
        }
    }

    /* 3. Navigation arrow and status. */
    draw_nav(fb, s, h);
    draw_status(fb, h);
}
