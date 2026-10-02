#include "nn_ref.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { OP_CONV = 1, OP_RELU, OP_RESIZE, OP_ADD, OP_SIGMOID };
#define NONE 0xFFFFFFFFu

/* Activations are float; the plain-JavaScript build (wasm2js) defines
 * NN_REF_F64 because JS engines run double arithmetic much faster than
 * emulated float32 (Math.fround on every operation). */
#ifdef NN_REF_F64
typedef double act_t;
#else
typedef float act_t;
#endif

typedef struct { uint32_t c, h, w; act_t *data; } tensor_t;
typedef struct {
    uint32_t type, in0, in1, out, kh, kw, sh, sw, pt, pl, pb, pr, scale_h, scale_w;
    const float *weights, *bias;
} op_t;

struct nn_ref {
    uint32_t n_tensors, n_ops, out_heat, out_wh, out_off;
    tensor_t *t;
    op_t *ops;
};

static uint32_t rd32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }

void nn_ref_free(nn_ref_t *m)
{
    if (!m) return;
    if (m->t) for (uint32_t i = 0; i < m->n_tensors; i++) free(m->t[i].data);
    free(m->t);
    free(m->ops);
    free(m);
}

nn_ref_t *nn_ref_load(const uint8_t *b, size_t len)
{
    if (len < 28 || memcmp(b, "PSNN", 4) || rd32(b + 4) != 1) return NULL;
    nn_ref_t *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->n_tensors = rd32(b + 8);
    m->n_ops = rd32(b + 12);
    m->out_heat = rd32(b + 16);
    m->out_wh = rd32(b + 20);
    m->out_off = rd32(b + 24);
    size_t off = 28;
    m->t = calloc(m->n_tensors, sizeof(tensor_t));
    m->ops = calloc(m->n_ops, sizeof(op_t));
    if (!m->t || !m->ops || off + 12u * m->n_tensors > len) goto fail;
    for (uint32_t i = 0; i < m->n_tensors; i++, off += 12) {
        tensor_t *t = &m->t[i];
        t->c = rd32(b + off); t->h = rd32(b + off + 4); t->w = rd32(b + off + 8);
        t->data = malloc(sizeof(act_t) * t->c * t->h * t->w);
        if (!t->data) goto fail;
    }
    for (uint32_t i = 0; i < m->n_ops; i++) {
        if (off + 56 > len) goto fail;
        op_t *o = &m->ops[i];
        uint32_t *f = &o->type;
        for (int k = 0; k < 14; k++) f[k] = rd32(b + off + 4 * k);
        off += 56;
        if (o->in0 >= m->n_tensors || o->out >= m->n_tensors || (o->in1 != NONE && o->in1 >= m->n_tensors)) goto fail;
        if (o->type == OP_CONV) {
            size_t nw = (size_t)m->t[o->out].c * m->t[o->in0].c * o->kh * o->kw;
            size_t need = 4 * (nw + m->t[o->out].c);
            if (off + need > len) goto fail;
            o->weights = (const float *)(b + off);
            o->bias = (const float *)(b + off + 4 * nw);
            off += need;
        }
    }
    return m;
fail:
    nn_ref_free(m);
    return NULL;
}

static void conv(const op_t *o, const tensor_t *in, tensor_t *out)
{
    const int IW = (int)in->w, IH = (int)in->h, OW = (int)out->w, OH = (int)out->h;
    const int KH = (int)o->kh, KW = (int)o->kw, SH = (int)o->sh, SW = (int)o->sw;
    const int PT = (int)o->pt, PL = (int)o->pl;
    for (uint32_t oc = 0; oc < out->c; oc++) {
        act_t *op = out->data + (size_t)oc * OH * OW;
        for (int i = 0; i < OH * OW; i++) op[i] = o->bias[oc];
        for (uint32_t ic = 0; ic < in->c; ic++) {
            const act_t *ip = in->data + (size_t)ic * IH * IW;
            const float *wk = o->weights + ((size_t)oc * in->c + ic) * KH * KW;
            for (int ky = 0; ky < KH; ky++)
                for (int kx = 0; kx < KW; kx++) {
                    const act_t wv = wk[ky * KW + kx];
                    /* Valid output columns: 0 <= ox*SW - PL + kx < IW */
                    int ox0 = 0;
                    while (ox0 < OW && ox0 * SW - PL + kx < 0) ox0++;
                    int ox1 = OW;
                    while (ox1 > ox0 && (ox1 - 1) * SW - PL + kx >= IW) ox1--;
                    for (int oy = 0; oy < OH; oy++) {
                        const int iy = oy * SH - PT + ky;
                        if (iy < 0 || iy >= IH) continue;
                        const act_t *irow = ip + (size_t)iy * IW - PL + kx;
                        act_t *orow = op + (size_t)oy * OW;
                        if (SW == 1) for (int ox = ox0; ox < ox1; ox++) orow[ox] += wv * irow[ox];
                        else for (int ox = ox0; ox < ox1; ox++) orow[ox] += wv * irow[ox * SW];
                    }
                }
        }
    }
}

int nn_ref_run(nn_ref_t *m, const uint8_t *codes, float *heat, float *wh, float *off)
{
    tensor_t *in = &m->t[0];
    if (in->c != 1 || in->h != PS_THERM_H || in->w != PS_THERM_W) return -1;
    for (int i = 0; i < PS_THERM_PIXELS; i++) in->data[i] = codes[i] * (act_t)(1.0 / 255.0);

    for (uint32_t k = 0; k < m->n_ops; k++) {
        const op_t *o = &m->ops[k];
        tensor_t *a = &m->t[o->in0], *y = &m->t[o->out];
        const size_t n = (size_t)y->c * y->h * y->w;
        switch (o->type) {
        case OP_CONV: conv(o, a, y); break;
        case OP_RELU: for (size_t i = 0; i < n; i++) y->data[i] = a->data[i] > 0 ? a->data[i] : 0; break;
        case OP_SIGMOID: for (size_t i = 0; i < n; i++) y->data[i] = (act_t)(1.0f / (1.0f + expf(-(float)a->data[i]))); break;
        case OP_ADD: {
            const tensor_t *bt = &m->t[o->in1];
            for (size_t i = 0; i < n; i++) y->data[i] = a->data[i] + bt->data[i];
            break;
        }
        case OP_RESIZE:
            for (uint32_t c = 0; c < y->c; c++)
                for (uint32_t yy = 0; yy < y->h; yy++)
                    for (uint32_t xx = 0; xx < y->w; xx++)
                        y->data[((size_t)c * y->h + yy) * y->w + xx] =
                            a->data[((size_t)c * a->h + yy / o->scale_h) * a->w + xx / o->scale_w];
            break;
        default: return -2;
        }
    }
    const size_t plane = 2u * 30 * 40;
    if (m->t[m->out_heat].c * m->t[m->out_heat].h * m->t[m->out_heat].w != plane) return -3;
    for (size_t i = 0; i < plane; i++) {
        heat[i] = (float)m->t[m->out_heat].data[i];
        wh[i] = (float)m->t[m->out_wh].data[i];
        off[i] = (float)m->t[m->out_off].data[i];
    }
    return 0;
}
