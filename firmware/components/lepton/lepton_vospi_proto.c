#include "lepton_vospi_proto.h"

#include <string.h>

#define WORDS_PER_PKT (LEP_PKT_PAYLOAD / 2)

uint16_t lep_crc16_update(uint16_t crc, uint8_t byte)
{
    crc ^= (uint16_t)byte << 8;
    for (int i = 0; i < 8; i++)
        crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    return crc;
}

uint16_t lep_crc16(const uint8_t *d, size_t n)
{
    uint16_t crc = 0;
    for (size_t i = 0; i < n; i++) crc = lep_crc16_update(crc, d[i]);
    return crc;
}

static uint16_t pkt_crc(const uint8_t *p)
{
    uint16_t crc = 0;
    crc = lep_crc16_update(crc, p[0] & 0x0F); /* ID with top nibble masked */
    crc = lep_crc16_update(crc, p[1]);
    crc = lep_crc16_update(crc, 0);           /* CRC field as zero */
    crc = lep_crc16_update(crc, 0);
    for (int i = 4; i < LEP_PKT_BYTES; i++) crc = lep_crc16_update(crc, p[i]);
    return crc;
}

bool lep_pkt_crc_ok(const uint8_t *p)
{
    return pkt_crc(p) == (uint16_t)((p[2] << 8) | p[3]);
}

void lep_pkt_set_crc(uint8_t *p)
{
    uint16_t c = pkt_crc(p);
    p[2] = (uint8_t)(c >> 8);
    p[3] = (uint8_t)c;
}

void lep_asm_init(lep_asm_t *a, bool check_crc)
{
    memset(a, 0, sizeof(*a));
    a->check_crc = check_crc;
    a->crc_autodisable = true;
    a->expected_seg = 1;
    a->seg_no = -1;
}

void lep_asm_resynced(lep_asm_t *a)
{
    if (a->frame_in_progress) a->c.frames_lost++;
    a->in_segment = false;
    a->expected_pkt = 0;
    a->seg_no = -1;
    a->expected_seg = 1;
    a->frame_in_progress = false;
    a->bad_streak = 0;
    a->need_resync = false;
    a->c.resyncs++;
}

static lep_feed_t fail(lep_asm_t *a)
{
    a->in_segment = false;
    a->expected_pkt = 0;
    a->seg_no = -1;
    if (++a->bad_streak >= LEP_BAD_STREAK_RESYNC) a->need_resync = true;
    return LEP_FEED_ERROR;
}

static void abandon_frame(lep_asm_t *a)
{
    if (a->frame_in_progress) a->c.frames_lost++;
    a->frame_in_progress = false;
    a->expected_seg = 1;
}

static lep_feed_t segment_done(lep_asm_t *a)
{
    const int s = a->seg_no;
    a->in_segment = false;
    a->expected_pkt = 0;
    a->seg_no = -1;
    a->bad_streak = 0;

    if (s == 0) {               /* repeated frame: neutral, does not break a frame */
        a->c.seg_invalid++;
        return LEP_FEED_SEGMENT;
    }
    if (s < 1 || s > LEP_SEGS_PER_FRAME) {
        a->c.seg_order_errors++;
        abandon_frame(a);
        return LEP_FEED_ERROR;
    }
    if (s == 1) {
        if (a->frame_in_progress) abandon_frame(a);
        a->frame_in_progress = true;
        a->expected_seg = 1;
    } else if (s != a->expected_seg || !a->frame_in_progress) {
        a->c.seg_order_errors++;
        abandon_frame(a);
        return LEP_FEED_ERROR;
    }
    memcpy(&a->raw[(s - 1) * LEP_ROWS_PER_SEG * PS_THERM_W], a->seg, sizeof(a->seg));
    a->c.segments++;
    if (s == LEP_SEGS_PER_FRAME) {
        a->frame_in_progress = false;
        a->expected_seg = 1;
        a->c.frames++;
        return LEP_FEED_FRAME;
    }
    a->expected_seg = s + 1;
    return LEP_FEED_SEGMENT;
}

lep_feed_t lep_asm_feed(lep_asm_t *a, const uint8_t *p)
{
    if (lep_pkt_is_discard(p)) {
        a->c.discards++;
        return LEP_FEED_DISCARD;
    }
    a->c.packets++;
    const int n = lep_pkt_number(p);

    if (a->check_crc) {
        if (!lep_pkt_crc_ok(p)) {
            a->c.crc_errors++;
            /* Safety net: a sensor that never passes the check but is otherwise
             * well-formed means our CRC variant is wrong; a blind eyepiece is
             * worse than an occasional corrupted packet. */
            if (a->crc_autodisable && a->crc_ok_ever == 0 && a->c.crc_errors >= 2000)
                a->check_crc = false;
            return fail(a);
        }
        a->crc_ok_ever++;
    }

    if (n >= LEP_PKTS_PER_SEG) {    /* includes telemetry rows if misconfigured */
        a->c.seq_errors++;
        return fail(a);
    }
    if (n == 0) {
        if (a->in_segment) a->c.seq_errors++;   /* restarted mid-segment */
        a->in_segment = true;
        a->expected_pkt = 0;
        a->seg_no = -1;
    } else if (!a->in_segment || n != a->expected_pkt) {
        if (a->in_segment) a->c.seq_errors++;
        return fail(a);
    }
    if (n == LEP_SEG_PKT_TTT) a->seg_no = lep_pkt_ttt(p);

    uint16_t *dst = &a->seg[n * WORDS_PER_PKT];
    const uint8_t *src = p + 4;
    for (int i = 0; i < WORDS_PER_PKT; i++) dst[i] = (uint16_t)((src[2 * i] << 8) | src[2 * i + 1]);

    a->expected_pkt = n + 1;
    if (n == LEP_PKTS_PER_SEG - 1) return segment_done(a);
    return LEP_FEED_NONE;
}

ps_dc_t lep_tlinear_to_dc(uint16_t raw, lep_tlinear_res_t res)
{
    /* deci-C = K*10 - 2731.5. Work in centi-Kelvin to keep the 0.15. */
    int32_t centi_c = (res == LEP_RES_0_1K ? (int32_t)raw * 10 : (int32_t)raw) - 27315;
    int32_t dc = centi_c >= 0 ? (centi_c + 5) / 10 : -((-centi_c + 5) / 10);
    if (dc > INT16_MAX) dc = INT16_MAX;
    if (dc < INT16_MIN) dc = INT16_MIN;
    return (ps_dc_t)dc;
}

void lep_asm_to_frame(const lep_asm_t *a, lep_tlinear_res_t res, ps_thermal_frame_t *out)
{
    /* Small LUT-free loop: 19200 conversions, integer only. */
    for (int i = 0; i < PS_THERM_PIXELS; i++) out->px[i] = lep_tlinear_to_dc(a->raw[i], res);
}
