/* Host tests: SHTP framing and SH-2 report parsing with synthetic byte streams. */
#include <math.h>
#include <string.h>

#include "../../core/tests/ps_test.h"
#include "pyrosight/ps_nav.h"
#include "shtp_proto.h"

#define MAXEV 32
static sh2_event_t evs[MAXEV];
static int nev;
static void cb(void *ctx, const sh2_event_t *e) { (void)ctx; if (nev < MAXEV) evs[nev++] = *e; }

static void put16(uint8_t *b, int v) { b[0] = (uint8_t)v; b[1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *b, uint32_t v) { for (int i = 0; i < 4; i++) b[i] = (uint8_t)(v >> (8 * i)); }

/* Report header: id, seq, status (accuracy + delay MSB), delay LSB. */
static size_t rep_hdr(uint8_t *b, uint8_t id, uint8_t seq, int acc, int delay)
{
    b[0] = id; b[1] = seq; b[2] = (uint8_t)(((delay >> 8) << 2) | acc); b[3] = (uint8_t)delay;
    return 4;
}

static size_t game_rv(uint8_t *b, float yaw, int delay)
{
    size_t n = rep_hdr(b, SH2_GAME_ROTATION_VECTOR, 1, 3, delay);
    float w = cosf(yaw / 2), z = sinf(yaw / 2);
    put16(b + n, 0); put16(b + n + 2, 0);
    put16(b + n + 4, (int)lroundf(z * 16384)); put16(b + n + 6, (int)lroundf(w * 16384));
    return 12;
}

static size_t lin_acc(uint8_t *b, float x, float y, float z)
{
    size_t n = rep_hdr(b, SH2_LINEAR_ACCELERATION, 2, 2, 0);
    put16(b + n, (int)lroundf(x * 256)); put16(b + n + 2, (int)lroundf(y * 256)); put16(b + n + 4, (int)lroundf(z * 256));
    return 10;
}

static size_t step_det(uint8_t *b, uint32_t latency)
{
    size_t n = rep_hdr(b, SH2_STEP_DETECTOR, 3, 0, 0);
    put32(b + n, latency);
    return 8;
}

static size_t base_ts(uint8_t *b, uint32_t delta)
{
    b[0] = SH2_BASE_TIMESTAMP; put32(b + 1, delta);
    return 5;
}

static void test_header_and_build(void)
{
    uint8_t out[64], pl[17];
    CHECK(sh2_build_set_feature(pl, SH2_GAME_ROTATION_VECTOR, 10000, 0, 0, 0) == 17);
    CHECK(pl[0] == 0xFD && pl[1] == 0x08 && pl[2] == 0);
    CHECK(pl[5] == 0x10 && pl[6] == 0x27 && pl[7] == 0 && pl[8] == 0);   /* 10000 us LE */
    size_t n = shtp_build_packet(out, sizeof(out), SHTP_CHAN_CONTROL, 7, pl, 17);
    CHECK(n == 21);
    CHECK(out[0] == 21 && out[1] == 0 && out[2] == 2 && out[3] == 7);
    shtp_hdr_t h;
    CHECK(shtp_parse_header(out, &h));
    CHECK(h.len == 21 && h.chan == 2 && h.seq == 7 && !h.cont);
    uint8_t cont[4] = { 0x10, 0x81, 3, 9 };
    CHECK(shtp_parse_header(cont, &h) && h.cont && h.len == 0x110);
    uint8_t idle[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    CHECK(!shtp_parse_header(idle, &h));
    uint8_t zero[4] = { 0, 0, 0, 0 };
    CHECK(!shtp_parse_header(zero, &h));
    uint8_t badch[4] = { 8, 0, 9, 0 };
    CHECK(!shtp_parse_header(badch, &h));
    CHECK(shtp_build_packet(out, 10, 2, 0, pl, 17) == 0);
    CHECK(sh2_build_product_id_request(pl) == 2 && pl[0] == 0xF9);
}

static void test_report_batch(void)
{
    sh2_parser_t p;
    sh2_parser_init(&p);
    uint8_t c[128];
    size_t n = 0;
    n += base_ts(c + n, 25);                 /* 2.5 ms before the interrupt */
    n += game_rv(c + n, 0.5f, 5);            /* +0.5 ms */
    n += lin_acc(c + n, 3.0f, -4.0f, 0.0f);
    n += step_det(c + n, 1234);
    uint8_t pkt[160];
    size_t pn = shtp_build_packet(pkt, sizeof(pkt), SHTP_CHAN_REPORTS, 0, c, n);
    nev = 0;
    CHECK(sh2_parse_packet(&p, pkt, pn, cb, NULL) == 3);
    CHECK(nev == 3);
    CHECK(evs[0].type == SH2_EV_GAME_RV);
    CHECK(evs[0].accuracy == 3);
    CHECK(evs[0].t_offset_us == -2000);
    CHECK_NEAR(ps_quat_to_yaw(evs[0].u.q.w, evs[0].u.q.x, evs[0].u.q.y, evs[0].u.q.z), 0.5, 1e-3);
    CHECK(evs[1].type == SH2_EV_LINEAR_ACCEL);
    CHECK_NEAR(sh2_vec_mag(&evs[1]), 5.0, 1e-2);
    CHECK(evs[2].type == SH2_EV_STEP && evs[2].u.step.latency_us == 1234);
    CHECK(p.reports == 4 && p.unknown == 0 && p.truncated == 0);

    /* Large delay uses the status MSBs. */
    n = 0;
    n += base_ts(c + n, 0);
    n += game_rv(c + n, -2.0f, 0x2A5);
    nev = 0;
    sh2_parse_cargo(&p, SHTP_CHAN_REPORTS, 1, c, n, cb, NULL);
    CHECK(nev == 1 && evs[0].t_offset_us == 0x2A5 * 100);
    CHECK_NEAR(ps_quat_to_yaw(evs[0].u.q.w, evs[0].u.q.x, evs[0].u.q.y, evs[0].u.q.z), -2.0, 1e-3);

    /* Rebase adds to the base. */
    n = 0;
    n += base_ts(c + n, 100);
    c[n] = SH2_TIMESTAMP_REBASE; put32(c + n + 1, 40); n += 5;
    n += step_det(c + n, 0);
    nev = 0;
    sh2_parse_cargo(&p, SHTP_CHAN_REPORTS, 2, c, n, cb, NULL);
    CHECK(nev == 1 && evs[0].t_offset_us == -6000);
    CHECK(p.seq_gaps == 0);
    sh2_parse_cargo(&p, SHTP_CHAN_REPORTS, 9, c, n, cb, NULL);
    CHECK(p.seq_gaps == 1);
}

static void test_unknown_and_truncated(void)
{
    sh2_parser_t p;
    sh2_parser_init(&p);
    uint8_t c[64];
    size_t n = 0;
    n += base_ts(c + n, 0);
    n += step_det(c + n, 1);
    c[n++] = 0x77; c[n++] = 0; c[n++] = 0;   /* unknown id: rest skipped */
    n += step_det(c + n, 2);
    nev = 0;
    CHECK(sh2_parse_cargo(&p, SHTP_CHAN_REPORTS, 0, c, n, cb, NULL) == 1);
    CHECK(p.unknown == 1);
    /* Truncated game RV at the end. */
    n = 0;
    n += step_det(c + n, 3);
    n += game_rv(c + n, 0.1f, 0);
    nev = 0;
    CHECK(sh2_parse_cargo(&p, SHTP_CHAN_REPORTS, 1, c, n - 3, cb, NULL) == 1);
    CHECK(p.truncated == 1);
    /* Unused but known reports are reported as OTHER and do not stop parsing. */
    n = 0;
    n += rep_hdr(c + n, SH2_ACCELEROMETER, 0, 0, 0); n += 6;
    n += step_det(c + n, 4);
    nev = 0;
    CHECK(sh2_parse_cargo(&p, SHTP_CHAN_REPORTS, 2, c, n, cb, NULL) == 2);
    CHECK(evs[0].type == SH2_EV_OTHER_REPORT && evs[1].type == SH2_EV_STEP);
}

static void test_reset_and_control(void)
{
    sh2_parser_t p;
    sh2_parser_init(&p);
    uint8_t pkt[64], c[32];
    /* Advertisement on channel 0. */
    uint8_t adv[] = { 0x00, 0x01, 0x04, 0x00, 0x00, 0x00, 0x01 };
    size_t pn = shtp_build_packet(pkt, sizeof(pkt), SHTP_CHAN_COMMAND, 0, adv, sizeof(adv));
    nev = 0;
    CHECK(sh2_parse_packet(&p, pkt, pn, cb, NULL) == 1 && evs[0].type == SH2_EV_ADVERTISEMENT);
    /* Reset complete on channel 1. */
    uint8_t rc[] = { SH2_EXEC_RESET_COMPLETE };
    pn = shtp_build_packet(pkt, sizeof(pkt), SHTP_CHAN_EXECUTABLE, 0, rc, 1);
    nev = 0;
    CHECK(sh2_parse_packet(&p, pkt, pn, cb, NULL) == 1 && evs[0].type == SH2_EV_RESET_COMPLETE);
    CHECK(p.resets == 1);
    /* Product ID response on channel 2. */
    memset(c, 0, sizeof(c));
    c[0] = SH2_PRODUCT_ID_RESPONSE; c[1] = 1; c[2] = 3; c[3] = 2;
    put32(c + 4, 10003608); put32(c + 8, 7); put16(c + 12, 17);
    pn = shtp_build_packet(pkt, sizeof(pkt), SHTP_CHAN_CONTROL, 0, c, 16);
    nev = 0;
    CHECK(sh2_parse_packet(&p, pkt, pn, cb, NULL) == 1);
    CHECK(evs[0].type == SH2_EV_PRODUCT_ID && evs[0].u.pid.sw_major == 3 &&
          evs[0].u.pid.part_no == 10003608 && evs[0].u.pid.patch == 17);
    /* Get feature response. */
    memset(c, 0, sizeof(c));
    c[0] = SH2_GET_FEATURE_RESPONSE; c[1] = SH2_GAME_ROTATION_VECTOR; put32(c + 5, 10000);
    nev = 0;
    CHECK(sh2_parse_cargo(&p, SHTP_CHAN_CONTROL, 1, c, 17, cb, NULL) == 1);
    CHECK(evs[0].type == SH2_EV_FEATURE_RESPONSE && evs[0].u.feature.report_id == 8 &&
          evs[0].u.feature.interval_us == 10000);
    /* Packet length shorter than header claims: parse what we have. */
    CHECK(sh2_parse_packet(&p, pkt, 3, cb, NULL) == -1);
}

int main(void)
{
    RUN(test_header_and_build);
    RUN(test_report_batch);
    RUN(test_unknown_and_truncated);
    RUN(test_reset_and_control);
    TEST_MAIN_END();
}
