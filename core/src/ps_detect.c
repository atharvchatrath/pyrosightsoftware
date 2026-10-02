#include "pyrosight/ps_detect.h"

#include <math.h>
#include <string.h>

#define MAX_LABELS 1024

typedef struct {
    int16_t x0, y0, x1, y1;
    uint16_t area;
    ps_dc_t peak;
} blob_t;

static uint16_t uf_find(uint16_t *parent, uint16_t a)
{
    while (parent[a] != a) {
        parent[a] = parent[parent[a]];
        a = parent[a];
    }
    return a;
}

static void uf_union(uint16_t *parent, uint16_t a, uint16_t b)
{
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a < b) parent[b] = a;
    else if (b < a) parent[a] = b;
}

/*
 * Two-pass 4-connected component labelling. On entry labels[] holds the
 * foreground mask (non-zero = foreground); on exit it holds labels.
 * Returns the number of blobs written to `blobs` (at most max_blobs, largest
 * kept if there are more).
 */
static int find_blobs(const ps_thermal_frame_t *f, uint16_t *labels, blob_t *blobs, int max_blobs)
{
    uint16_t parent[MAX_LABELS];
    uint16_t next = 1;
    parent[0] = 0;

    for (int y = 0; y < PS_THERM_H; y++) {
        for (int x = 0; x < PS_THERM_W; x++) {
            const int i = y * PS_THERM_W + x;
            if (!labels[i]) continue;
            uint16_t up = y > 0 ? labels[i - PS_THERM_W] : 0;
            uint16_t left = x > 0 ? labels[i - 1] : 0;
            if (!up && !left) {
                if (next < MAX_LABELS) { parent[next] = next; labels[i] = next++; }
                else labels[i] = 0; /* label table full: drop the pixel */
            } else if (up && left) {
                labels[i] = up < left ? up : left;
                if (up != left) uf_union(parent, up, left);
            } else {
                labels[i] = up ? up : left;
            }
        }
    }

    /* Resolve and accumulate per root. Reuse parent[] roots as indices into a
     * compact table. */
    static blob_t acc[MAX_LABELS]; /* static: 16 KB would blow a small task stack */
    uint16_t compact[MAX_LABELS];
    int nroots = 0;
    for (uint16_t l = 1; l < next; l++) {
        if (uf_find(parent, l) == l) {
            compact[l] = (uint16_t)nroots;
            acc[nroots] = (blob_t){ .x0 = 32767, .y0 = 32767, .x1 = -1, .y1 = -1, .area = 0, .peak = INT16_MIN };
            nroots++;
        }
    }
    for (int y = 0; y < PS_THERM_H; y++) {
        for (int x = 0; x < PS_THERM_W; x++) {
            const int i = y * PS_THERM_W + x;
            if (!labels[i]) continue;
            blob_t *b = &acc[compact[uf_find(parent, labels[i])]];
            if (x < b->x0) b->x0 = (int16_t)x;
            if (y < b->y0) b->y0 = (int16_t)y;
            if (x > b->x1) b->x1 = (int16_t)x;
            if (y > b->y1) b->y1 = (int16_t)y;
            if (b->area < UINT16_MAX) b->area++;
            if (f->px[i] > b->peak) b->peak = f->px[i];
        }
    }

    /* Keep the largest max_blobs (partial selection sort; nroots is small). */
    int n = 0;
    for (int k = 0; k < max_blobs && k < nroots; k++) {
        int best = -1;
        for (int j = 0; j < nroots; j++)
            if (acc[j].area && (best < 0 || acc[j].area > acc[best].area)) best = j;
        if (best < 0) break;
        blobs[n++] = acc[best];
        acc[best].area = 0;
    }
    return n;
}

static void push(ps_detections_t *out, ps_class_t cls, float score, const blob_t *b)
{
    if (out->n >= PS_MAX_DETECTIONS) return;
    ps_detection_t *d = &out->d[out->n++];
    memset(d, 0, sizeof(*d));
    d->cls = cls;
    d->score = score;
    d->x = b->x0;
    d->y = b->y0;
    d->w = (float)(b->x1 - b->x0 + 1);
    d->h = (float)(b->y1 - b->y0 + 1);
    d->peak_dc = b->peak;
}

void ps_detect_classical(const ps_config_t *cfg, const ps_thermal_frame_t *f,
                         uint16_t *labels, ps_detections_t *out)
{
    blob_t blobs[PS_MAX_DETECTIONS];
    out->n = 0;
    out->frame_id = f->frame_id;
    out->t_ms = f->t_ms;
    out->source = PS_DETECTOR_CLASSICAL;

    /* Fire: anything hotter than the flame threshold. */
    for (int i = 0; i < PS_THERM_PIXELS; i++) labels[i] = f->px[i] >= cfg->fire_min_dc;
    int n = find_blobs(f, labels, blobs, 8);
    for (int i = 0; i < n; i++) {
        if (blobs[i].area < cfg->fire_min_area) continue;
        float s = 0.5f + 0.5f * fminf(1.0f, (float)blobs[i].area / 200.0f);
        push(out, PS_CLASS_FIRE, s, &blobs[i]);
    }

    /* People: surface temperature band AND a local bump in the horizontal
     * temperature profile. In a fire room the walls at head height sit inside
     * the human band and a person can be cooler than the hot gas around them,
     * so absolute thresholds alone flood the frame. A person is warmer (or
     * cooler) than BOTH sides at some offset; a wall seen in perspective is a
     * monotonic gradient and fails one side. Two offsets cover near and far. */
    static const int offs[2] = { 8, 22 };
    const int c = cfg->person_contrast_dc;
    for (int y = 0; y < PS_THERM_H; y++) {
        const ps_dc_t *row = &f->px[y * PS_THERM_W];
        for (int x = 0; x < PS_THERM_W; x++) {
            int v = row[x];
            bool fg = false;
            if (v >= cfg->person_min_dc && v <= cfg->person_max_dc) {
                for (int k = 0; k < 2 && !fg; k++) {
                    int xl = x - offs[k], xr = x + offs[k];
                    if (xl < 0 || xr >= PS_THERM_W) continue;
                    int dl = v - row[xl], dr = v - row[xr];
                    fg = (dl > c && dr > c) || (dl < -c && dr < -c);
                }
            }
            labels[y * PS_THERM_W + x] = fg;
        }
    }
    /* Erode with a 4-neighbour cross: removes single-pixel noise and the thin
     * diagonal strips where walls meet the floor. Boxes are re-grown below. */
    for (int y = 0; y < PS_THERM_H; y++)
        for (int x = 0; x < PS_THERM_W; x++) {
            int i = y * PS_THERM_W + x;
            bool keep = labels[i] && x > 0 && y > 0 && x < PS_THERM_W - 1 && y < PS_THERM_H - 1 &&
                        labels[i - 1] && labels[i + 1] && labels[i - PS_THERM_W] && labels[i + PS_THERM_W];
            labels[i] = keep ? 1 : (labels[i] ? 2 : 0); /* 2 = eroded away, still visible below */
        }
    for (int i = 0; i < PS_THERM_PIXELS; i++) labels[i] = labels[i] == 1;
    n = find_blobs(f, labels, blobs, PS_MAX_DETECTIONS);
    /* A person seen against a hot gas layer often splits into head, torso and
     * legs (the torso can match the wall behind it). Merge fragments stacked
     * in the same columns with a small vertical gap. */
    for (bool merged = true; merged;) {
        merged = false;
        for (int i = 0; i < n && !merged; i++)
            for (int j = i + 1; j < n && !merged; j++) {
                blob_t *a = &blobs[i], *b = &blobs[j];
                int ox = (a->x1 < b->x1 ? a->x1 : b->x1) - (a->x0 > b->x0 ? a->x0 : b->x0) + 1;
                int wa = a->x1 - a->x0 + 1, wb = b->x1 - b->x0 + 1;
                int gap = a->y0 > b->y1 ? a->y0 - b->y1 : (b->y0 > a->y1 ? b->y0 - a->y1 : 0);
                int hmax = (a->y1 - a->y0 > b->y1 - b->y0 ? a->y1 - a->y0 : b->y1 - b->y0) + 1;
                if (ox * 2 >= (wa < wb ? wa : wb) && gap <= hmax / 2 + 2) {
                    if (b->x0 < a->x0) a->x0 = b->x0;
                    if (b->y0 < a->y0) a->y0 = b->y0;
                    if (b->x1 > a->x1) a->x1 = b->x1;
                    if (b->y1 > a->y1) a->y1 = b->y1;
                    a->area = (uint16_t)(a->area + b->area);
                    if (b->peak > a->peak) a->peak = b->peak;
                    blobs[j] = blobs[--n];
                    merged = true;
                }
            }
    }
    for (int i = 0; i < n; i++) {
        blob_t g = blobs[i];
        /* Undo the erosion on the box. */
        if (g.x0 > 0) g.x0--;
        if (g.y0 > 0) g.y0--;
        if (g.x1 < PS_THERM_W - 1) g.x1++;
        if (g.y1 < PS_THERM_H - 1) g.y1++;
        const blob_t *b = &g;
        if (b->area < cfg->person_min_area) continue;
        if (b->area > PS_THERM_PIXELS / 3) continue; /* background in a warm room */
        float w = (float)(b->x1 - b->x0 + 1), h = (float)(b->y1 - b->y0 + 1);
        float aspect = h / w;                        /* standing ~3, lying ~0.35 */
        if (aspect < 0.2f || aspect > 6.0f) continue;
        float fill = (float)b->area / (w * h);
        if (fill < 0.40f) continue; /* people are compact; wall/floor seams are not */
        bool upright = aspect > 1.4f, prone = aspect < 0.7f;
        float s = 0.3f + 0.4f * fminf(1.0f, fill / 0.6f) + ((upright || prone) ? 0.2f : 0.0f);
        push(out, PS_CLASS_PERSON, s, b);
    }

    ps_nms(out, cfg->nms_iou);
}

float ps_iou(const ps_detection_t *a, const ps_detection_t *b)
{
    float x0 = fmaxf(a->x, b->x), y0 = fmaxf(a->y, b->y);
    float x1 = fminf(a->x + a->w, b->x + b->w), y1 = fminf(a->y + a->h, b->y + b->h);
    float iw = x1 - x0, ih = y1 - y0;
    if (iw <= 0 || ih <= 0) return 0.0f;
    float inter = iw * ih;
    return inter / (a->w * a->h + b->w * b->h - inter);
}

void ps_nms(ps_detections_t *dets, float iou_threshold)
{
    /* Insertion sort by score, descending. */
    for (int i = 1; i < dets->n; i++) {
        ps_detection_t key = dets->d[i];
        int j = i - 1;
        while (j >= 0 && dets->d[j].score < key.score) { dets->d[j + 1] = dets->d[j]; j--; }
        dets->d[j + 1] = key;
    }
    bool keep[PS_MAX_DETECTIONS];
    for (int i = 0; i < dets->n; i++) keep[i] = true;
    for (int i = 0; i < dets->n; i++) {
        if (!keep[i]) continue;
        for (int j = i + 1; j < dets->n; j++)
            if (keep[j] && dets->d[j].cls == dets->d[i].cls &&
                ps_iou(&dets->d[i], &dets->d[j]) > iou_threshold)
                keep[j] = false;
    }
    int w = 0;
    for (int i = 0; i < dets->n; i++) if (keep[i]) dets->d[w++] = dets->d[i];
    dets->n = (uint8_t)w;
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

void ps_centernet_decode(const float *heat, const float *wh, const float *off,
                         int gw, int gh, int stride, float thr, ps_detections_t *out)
{
    const int plane = gw * gh;
    out->n = 0;
    out->source = PS_DETECTOR_NEURAL;
    /* Compare logits (monotonic with sigmoid) to avoid exp() on every cell. */
    const float thr_logit = logf(thr / (1.0f - thr));
    for (int c = 0; c < PS_NUM_CLASSES; c++) {
        const float *hm = heat + c * plane;
        for (int gy = 0; gy < gh; gy++) {
            for (int gx = 0; gx < gw; gx++) {
                const float v = hm[gy * gw + gx];
                if (v < thr_logit) continue;
                bool is_max = true;
                for (int dy = -1; dy <= 1 && is_max; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int yy = gy + dy, xx = gx + dx;
                        if ((dx || dy) && yy >= 0 && yy < gh && xx >= 0 && xx < gw &&
                            hm[yy * gw + xx] > v) { is_max = false; break; }
                    }
                if (!is_max) continue;
                const int k = gy * gw + gx;
                float cx = (gx + off[k]) * stride, cy = (gy + off[plane + k]) * stride;
                float w = wh[k], h = wh[plane + k];
                if (w < 1.0f || h < 1.0f) continue;
                ps_detection_t d;
                memset(&d, 0, sizeof(d));
                d.cls = (ps_class_t)c;
                d.score = sigmoidf(v);
                d.x = cx - w * 0.5f;
                d.y = cy - h * 0.5f;
                d.w = w;
                d.h = h;
                if (out->n < PS_MAX_DETECTIONS) {
                    out->d[out->n++] = d;
                } else {
                    /* Replace the weakest if this one is stronger. */
                    int weakest = 0;
                    for (int i = 1; i < out->n; i++) if (out->d[i].score < out->d[weakest].score) weakest = i;
                    if (out->d[weakest].score < d.score) out->d[weakest] = d;
                }
            }
        }
    }
    ps_nms(out, 0.5f);
}

void ps_detect_fill_peaks(const ps_thermal_frame_t *f, ps_detections_t *dets)
{
    for (int i = 0; i < dets->n; i++) {
        ps_detection_t *d = &dets->d[i];
        int x0 = (int)fmaxf(0, d->x), y0 = (int)fmaxf(0, d->y);
        int x1 = (int)fminf(PS_THERM_W - 1, d->x + d->w - 1);
        int y1 = (int)fminf(PS_THERM_H - 1, d->y + d->h - 1);
        ps_dc_t pk = INT16_MIN;
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++)
                if (f->px[y * PS_THERM_W + x] > pk) pk = f->px[y * PS_THERM_W + x];
        d->peak_dc = pk;
    }
}

void ps_estimate_distances(const ps_config_t *cfg, ps_detections_t *dets)
{
    /* Pinhole: focal length in pixels from the horizontal field of view.
     * Lepton 3.5 pixels are square, so the same f applies vertically. */
    const float f_px = (PS_THERM_W * 0.5f) / tanf(cfg->hfov_deg * 0.5f * 3.14159265f / 180.0f);
    for (int i = 0; i < dets->n; i++) {
        ps_detection_t *d = &dets->d[i];
        d->truncated = d->x <= 0.5f || d->y <= 0.5f ||
                       d->x + d->w >= PS_THERM_W - 0.5f || d->y + d->h >= PS_THERM_H - 0.5f;
        if (d->cls != PS_CLASS_PERSON) {
            d->dist_m = d->dist_min_m = d->dist_max_m = 0.0f;
            continue;
        }
        /* A person's longest box side is roughly body length whether standing,
         * lying or crawling side-on. Crouching/kneeling and foreshortening make
         * the true size smaller, so the real distance is usually <= estimate. */
        float px = fmaxf(d->w, d->h);
        if (px < 1.0f) px = 1.0f;
        float est = f_px * cfg->person_height_m / px;
        d->dist_m = est;
        d->dist_min_m = est * 0.55f; /* crouched (~0.95 m) or end-on */
        d->dist_max_m = est * 1.15f; /* tall person / helmet */
        if (d->truncated) d->dist_min_m = 0.3f; /* partly out of frame: closer than it looks */
    }
}
