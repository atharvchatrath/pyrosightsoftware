#include "audio_pack.h"

#include <math.h>
#include <string.h>

static uint16_t rd16(const uint8_t *b) { return (uint16_t)(b[0] | (b[1] << 8)); }
static uint32_t rd32(const uint8_t *b) { return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24); }

uint32_t audio_crc32(const uint8_t *d, size_t n)
{
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        c ^= d[i];
        for (int k = 0; k < 8; k++) c = (c & 1) ? (c >> 1) ^ 0xEDB88320u : c >> 1;
    }
    return ~c;
}

uint16_t audio_pack_peek_count(const uint8_t *h, size_t n)
{
    if (n < AUDIO_PACK_HDR_BYTES || memcmp(h, AUDIO_PACK_MAGIC, 4) != 0) return 0;
    return rd16(h + 6);
}

int audio_pack_parse(const uint8_t *b, size_t n, size_t part_size, audio_pack_t *out)
{
    memset(out, 0, sizeof(*out));
    if (n < AUDIO_PACK_HDR_BYTES) return AUDIO_PACK_ERR_SHORT;
    if (memcmp(b, AUDIO_PACK_MAGIC, 4) != 0) return AUDIO_PACK_ERR_MAGIC;
    if (rd16(b + 4) != AUDIO_PACK_VERSION) return AUDIO_PACK_ERR_VERSION;
    const uint16_t count = rd16(b + 6);
    const uint32_t rate = rd32(b + 8);
    if (rate != AUDIO_SAMPLE_RATE || b[12] != 16 || b[13] != 1) return AUDIO_PACK_ERR_FORMAT;
    if (count == 0 || count > AUDIO_PACK_MAX_ENTRIES) return AUDIO_PACK_ERR_COUNT;
    const size_t idx_end = audio_pack_index_bytes(count);
    if (n < idx_end) return AUDIO_PACK_ERR_SHORT;
    const uint32_t image_size = rd32(b + 16);
    if (image_size > part_size || image_size < idx_end) return AUDIO_PACK_ERR_BOUNDS;
    if (audio_crc32(b + AUDIO_PACK_HDR_BYTES, idx_end - AUDIO_PACK_HDR_BYTES) != rd32(b + 20))
        return AUDIO_PACK_ERR_CRC;

    out->count = count;
    out->sample_rate = rate;
    out->image_size = image_size;
    for (uint16_t i = 0; i < count; i++) {
        const uint8_t *e = b + AUDIO_PACK_HDR_BYTES + i * AUDIO_PACK_ENTRY_BYTES;
        uint32_t off = rd32(e), len = rd32(e + 4);
        bool ok = len > 0 && (len & 1) == 0 && off >= idx_end &&
                  (uint64_t)off + len <= image_size;
        if (ok) {
            out->clip[i].offset = off;
            out->clip[i].length = len;
            out->n_valid++;
        }
    }
    return AUDIO_PACK_OK;
}

bool audio_pack_clip(const audio_pack_t *p, unsigned id, uint32_t *offset, uint32_t *length)
{
    if (!p || id >= p->count || p->clip[id].length == 0) return false;
    if (offset) *offset = p->clip[id].offset;
    if (length) *length = p->clip[id].length;
    return true;
}

int audio_beep_pattern(int prio, audio_tone_t o[AUDIO_BEEP_MAX_TONES])
{
    int n = 0;
    switch (prio) {
    case 0:  /* info: one soft short beep */
        o[n++] = (audio_tone_t){ 880, 120 };
        break;
    case 1:  /* navigation: two beeps */
        o[n++] = (audio_tone_t){ 988, 120 };
        o[n++] = (audio_tone_t){ 0, 80 };
        o[n++] = (audio_tone_t){ 988, 120 };
        break;
    case 2:  /* warning: three rising beeps */
        o[n++] = (audio_tone_t){ 1047, 100 };
        o[n++] = (audio_tone_t){ 0, 60 };
        o[n++] = (audio_tone_t){ 1319, 100 };
        o[n++] = (audio_tone_t){ 0, 60 };
        o[n++] = (audio_tone_t){ 1568, 160 };
        break;
    default: /* critical: fast two-tone alarm */
        for (int i = 0; i < 4; i++) {
            o[n++] = (audio_tone_t){ 1760, 90 };
            o[n++] = (audio_tone_t){ 1175, 90 };
        }
        o[n++] = (audio_tone_t){ 0, 40 };
        break;
    }
    return n;
}

void audio_tone_gen_init(audio_tone_gen_t *g, const audio_tone_t *tones, int n, int16_t amplitude)
{
    memset(g, 0, sizeof(*g));
    g->tones = tones;
    g->n_tones = n;
    g->amplitude = amplitude;
}

size_t audio_tone_gen_fill(audio_tone_gen_t *g, int16_t *buf, size_t n)
{
    size_t w = 0;
    const float two_pi = 6.28318530718f;
    while (w < n && g->idx < g->n_tones) {
        const audio_tone_t *t = &g->tones[g->idx];
        const uint32_t len = (uint32_t)t->ms * AUDIO_SAMPLE_RATE / 1000;
        const uint32_t ramp = AUDIO_SAMPLE_RATE / 200;   /* 5 ms fade in/out: no clicks */
        while (w < n && g->pos < len) {
            int16_t s = 0;
            if (t->freq_hz) {
                float env = 1.0f;
                if (g->pos < ramp) env = (float)g->pos / ramp;
                else if (len - g->pos < ramp) env = (float)(len - g->pos) / ramp;
                s = (int16_t)(g->amplitude * env * sinf(g->phase));
                g->phase += two_pi * t->freq_hz / AUDIO_SAMPLE_RATE;
                if (g->phase > two_pi) g->phase -= two_pi;
            }
            buf[w++] = s;
            g->pos++;
        }
        if (g->pos >= len) { g->idx++; g->pos = 0; g->phase = 0; }
    }
    return w;
}
