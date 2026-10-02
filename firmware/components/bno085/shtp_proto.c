#include "shtp_proto.h"

#include <math.h>
#include <string.h>

static uint16_t rd16(const uint8_t *b) { return (uint16_t)(b[0] | (b[1] << 8)); }
static uint32_t rd32(const uint8_t *b) { return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24); }
static void wr32(uint8_t *b, uint32_t v) { b[0] = (uint8_t)v; b[1] = (uint8_t)(v >> 8); b[2] = (uint8_t)(v >> 16); b[3] = (uint8_t)(v >> 24); }

bool shtp_parse_header(const uint8_t *b, shtp_hdr_t *h)
{
    uint16_t raw = rd16(b);
    if (raw == 0xFFFF) return false;          /* bus idle / no device */
    h->cont = (raw & 0x8000) != 0;
    h->len = raw & 0x7FFF;
    h->chan = b[2];
    h->seq = b[3];
    if (h->len < SHTP_HDR_LEN || h->chan >= SHTP_NUM_CHANNELS) return false;
    return true;
}

size_t shtp_build_packet(uint8_t *out, size_t cap, uint8_t chan, uint8_t seq, const uint8_t *payload, size_t n)
{
    size_t len = n + SHTP_HDR_LEN;
    if (len > cap || len > 0x7FFF) return 0;
    out[0] = (uint8_t)len;
    out[1] = (uint8_t)(len >> 8);
    out[2] = chan;
    out[3] = seq;
    if (n) memcpy(out + SHTP_HDR_LEN, payload, n);
    return len;
}

size_t sh2_build_set_feature(uint8_t out[17], uint8_t report_id, uint32_t interval_us,
                             uint32_t batch_us, uint16_t sensitivity, uint32_t specific)
{
    out[0] = SH2_SET_FEATURE_COMMAND;
    out[1] = report_id;
    out[2] = 0;                       /* feature flags: no change-sensitivity, no wake */
    out[3] = (uint8_t)sensitivity;
    out[4] = (uint8_t)(sensitivity >> 8);
    wr32(&out[5], interval_us);
    wr32(&out[9], batch_us);
    wr32(&out[13], specific);
    return 17;
}

size_t sh2_build_product_id_request(uint8_t out[2])
{
    out[0] = SH2_PRODUCT_ID_REQUEST;
    out[1] = 0;
    return 2;
}

int sh2_report_len(uint8_t id)
{
    switch (id) {
    case 0x01: case 0x02: case 0x03: case 0x04: case 0x06: return 10;  /* accel, gyro, mag, lin acc, gravity */
    case 0x05: case 0x09: case 0x28: case 0x2A: return 14;              /* RV, geomag RV, ARVR RV, gyro-int RV */
    case 0x07: case 0x0F: return 16;                                    /* uncalibrated gyro / mag */
    case 0x08: case 0x29: return 12;                                    /* game RV, ARVR game RV */
    case 0x0A: case 0x0B: return 8;                                     /* pressure, ambient light */
    case 0x0C: case 0x0D: case 0x0E: return 6;                          /* humidity, proximity, temperature */
    case 0x10: return 5;                                                /* tap */
    case 0x11: return 12;                                               /* step counter */
    case 0x12: case 0x13: return 6;                                     /* significant motion, stability classifier */
    case 0x14: case 0x15: case 0x16: return 16;                         /* raw accel / gyro / mag */
    case 0x18: return 8;                                                /* step detector */
    case 0x19: case 0x1A: case 0x1B: case 0x1C: return 6;               /* shake, flip, pickup, stability det. */
    case 0x1E: return 16;                                               /* personal activity classifier */
    case 0x1F: case 0x20: case 0x21: case 0x22: case 0x23: return 6;    /* sleep, tilt, pocket, circle, heart */
    case SH2_CMD_RESPONSE: return 16;
    case SH2_PRODUCT_ID_RESPONSE: return 16;
    case SH2_TIMESTAMP_REBASE: return 5;
    case SH2_BASE_TIMESTAMP: return 5;
    case SH2_GET_FEATURE_RESPONSE: return 17;
    default: return -1;
    }
}

void sh2_parser_init(sh2_parser_t *p) { memset(p, 0, sizeof(*p)); }

static float q_to_f(const uint8_t *b, int q) { return (float)(int16_t)rd16(b) / (float)(1 << q); }

float sh2_vec_mag(const sh2_event_t *ev)
{
    return sqrtf(ev->u.v.x * ev->u.v.x + ev->u.v.y * ev->u.v.y + ev->u.v.z * ev->u.v.z);
}

static int parse_reports(sh2_parser_t *p, const uint8_t *c, size_t n, sh2_event_cb_t cb, void *ctx)
{
    int events = 0;
    int32_t base_100us = 0;  /* sample time base relative to the interrupt */
    size_t i = 0;
    while (i < n) {
        const uint8_t id = c[i];
        const int len = sh2_report_len(id);
        if (len < 0) { p->unknown++; break; }           /* cannot know where the next one starts */
        if (i + (size_t)len > n) { p->truncated++; break; }
        const uint8_t *r = c + i;
        i += (size_t)len;
        p->reports++;

        if (id == SH2_BASE_TIMESTAMP) { base_100us = -(int32_t)rd32(r + 1); continue; }
        if (id == SH2_TIMESTAMP_REBASE) { base_100us += (int32_t)rd32(r + 1); continue; }

        sh2_event_t ev;
        memset(&ev, 0, sizeof(ev));
        ev.report_id = id;
        if (id < 0xF0) {
            ev.accuracy = r[2] & 0x03;
            const int32_t delay = ((r[2] & 0xFC) << 6) | r[3];   /* 14 bits, 100 us units */
            ev.t_offset_us = (base_100us + delay) * 100;
        }
        switch (id) {
        case SH2_GAME_ROTATION_VECTOR:
            ev.type = SH2_EV_GAME_RV;
            ev.u.q.x = q_to_f(r + 4, SH2_Q_GAME_RV);
            ev.u.q.y = q_to_f(r + 6, SH2_Q_GAME_RV);
            ev.u.q.z = q_to_f(r + 8, SH2_Q_GAME_RV);
            ev.u.q.w = q_to_f(r + 10, SH2_Q_GAME_RV);
            break;
        case SH2_LINEAR_ACCELERATION:
            ev.type = SH2_EV_LINEAR_ACCEL;
            ev.u.v.x = q_to_f(r + 4, SH2_Q_LINEAR_ACC);
            ev.u.v.y = q_to_f(r + 6, SH2_Q_LINEAR_ACC);
            ev.u.v.z = q_to_f(r + 8, SH2_Q_LINEAR_ACC);
            break;
        case SH2_STEP_DETECTOR:
            ev.type = SH2_EV_STEP;
            ev.u.step.latency_us = rd32(r + 4);
            break;
        case SH2_GET_FEATURE_RESPONSE:
            ev.type = SH2_EV_FEATURE_RESPONSE;
            ev.u.feature.report_id = r[1];
            ev.u.feature.flags = r[2];
            ev.u.feature.interval_us = rd32(r + 5);
            break;
        case SH2_PRODUCT_ID_RESPONSE:
            ev.type = SH2_EV_PRODUCT_ID;
            ev.u.pid.reset_cause = r[1];
            ev.u.pid.sw_major = r[2];
            ev.u.pid.sw_minor = r[3];
            ev.u.pid.part_no = rd32(r + 4);
            ev.u.pid.build = rd32(r + 8);
            ev.u.pid.patch = rd16(r + 12);
            break;
        default:
            ev.type = SH2_EV_OTHER_REPORT;
            break;
        }
        if (cb) cb(ctx, &ev);
        events++;
    }
    return events;
}

int sh2_parse_cargo(sh2_parser_t *p, uint8_t chan, uint8_t seq, const uint8_t *c, size_t n,
                    sh2_event_cb_t cb, void *ctx)
{
    p->packets++;
    if (chan < SHTP_NUM_CHANNELS) {
        if (chan == SHTP_CHAN_REPORTS && p->seq_valid[chan] && (uint8_t)(p->last_seq[chan] + 1) != seq)
            p->seq_gaps++;
        p->last_seq[chan] = seq;
        p->seq_valid[chan] = true;
    }
    sh2_event_t ev;
    memset(&ev, 0, sizeof(ev));
    switch (chan) {
    case SHTP_CHAN_COMMAND:
        /* Advertisement (tag/length/value) starts with command 0x00. */
        if (n >= 1 && c[0] == 0x00) {
            ev.type = SH2_EV_ADVERTISEMENT;
            if (cb) cb(ctx, &ev);
            return 1;
        }
        return 0;
    case SHTP_CHAN_EXECUTABLE:
        if (n >= 1 && c[0] == SH2_EXEC_RESET_COMPLETE) {
            p->resets++;
            /* Sequence numbers restart after a reset. */
            memset(p->seq_valid, 0, sizeof(p->seq_valid));
            ev.type = SH2_EV_RESET_COMPLETE;
            if (cb) cb(ctx, &ev);
            return 1;
        }
        return 0;
    case SHTP_CHAN_CONTROL:
    case SHTP_CHAN_REPORTS:
    case SHTP_CHAN_WAKE:
        return parse_reports(p, c, n, cb, ctx);
    default:
        return 0;   /* gyro-integrated RV channel: not enabled */
    }
}

int sh2_parse_packet(sh2_parser_t *p, const uint8_t *pkt, size_t n, sh2_event_cb_t cb, void *ctx)
{
    shtp_hdr_t h;
    if (n < SHTP_HDR_LEN || !shtp_parse_header(pkt, &h)) return -1;
    size_t len = h.len < n ? h.len : n;
    return sh2_parse_cargo(p, h.chan, h.seq, pkt + SHTP_HDR_LEN, len - SHTP_HDR_LEN, cb, ctx);
}
