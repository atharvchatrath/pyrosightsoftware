/* Host tests: VoSPI packet/segment/frame assembly, CRC, TLinear, CCI sequencing. */
#include <string.h>

#include "../../core/tests/ps_test.h"
#include "lepton_cci.h"
#include "lepton_vospi_proto.h"

static lep_asm_t A;

/* Pixel value encodes (row, col) so we can check placement. */
static uint16_t pix(int row, int col, int salt) { return (uint16_t)(2731 + row * 7 + col + salt); }

static void make_pkt(uint8_t *p, int seg, int pkt, int salt)
{
    memset(p, 0, LEP_PKT_BYTES);
    int ttt = (pkt == LEP_SEG_PKT_TTT) ? seg : 0;
    uint16_t id = (uint16_t)((ttt << 12) | pkt);
    p[0] = (uint8_t)(id >> 8);
    p[1] = (uint8_t)id;
    int seg_idx = seg >= 1 ? seg - 1 : 0;
    int row = seg_idx * LEP_ROWS_PER_SEG + pkt / 2;
    int col0 = (pkt & 1) * 80;
    for (int i = 0; i < 80; i++) {
        uint16_t v = pix(row, col0 + i, salt);
        p[4 + 2 * i] = (uint8_t)(v >> 8);
        p[5 + 2 * i] = (uint8_t)v;
    }
    lep_pkt_set_crc(p);
}

static void make_discard(uint8_t *p)
{
    memset(p, 0xAB, LEP_PKT_BYTES);
    p[0] = 0x0F; p[1] = 0x00 | 0x3C;  /* xFxx */
}

/* Feed a whole segment; returns the result of the last packet. */
static lep_feed_t feed_seg(int seg, int salt)
{
    uint8_t p[LEP_PKT_BYTES];
    lep_feed_t r = LEP_FEED_NONE;
    for (int k = 0; k < LEP_PKTS_PER_SEG; k++) {
        make_pkt(p, seg, k, salt);
        r = lep_asm_feed(&A, p);
        if (k < LEP_PKTS_PER_SEG - 1) CHECK(r == LEP_FEED_NONE);
    }
    return r;
}

static void check_frame(int salt)
{
    int bad = 0;
    for (int r = 0; r < 120; r++)
        for (int c = 0; c < 160; c++)
            if (A.raw[r * 160 + c] != pix(r, c, salt)) bad++;
    CHECK(bad == 0);
}

static void test_crc(void)
{
    /* CRC-16/XMODEM check value. */
    CHECK(lep_crc16((const uint8_t *)"123456789", 9) == 0x31C3);
    uint8_t p[LEP_PKT_BYTES];
    make_pkt(p, 1, 20, 0);   /* TTT bits set: masked out of the CRC */
    CHECK(lep_pkt_crc_ok(p));
    uint8_t q[LEP_PKT_BYTES];
    memcpy(q, p, sizeof(q));
    q[0] = (uint8_t)((q[0] & 0x0F) | 0x30);  /* different TTT, same CRC */
    CHECK(lep_pkt_crc_ok(q));
    p[100] ^= 0x01;
    CHECK(!lep_pkt_crc_ok(p));
    CHECK(lep_pkt_ttt(q) == 3);
    CHECK(lep_pkt_number(q) == 20);
}

static void test_full_frame_with_discards(void)
{
    lep_asm_init(&A, true);
    uint8_t d[LEP_PKT_BYTES];
    make_discard(d);
    for (int i = 0; i < 5; i++) CHECK(lep_asm_feed(&A, d) == LEP_FEED_DISCARD);
    CHECK(feed_seg(1, 0) == LEP_FEED_SEGMENT);
    CHECK(lep_asm_feed(&A, d) == LEP_FEED_DISCARD);
    CHECK(feed_seg(2, 0) == LEP_FEED_SEGMENT);
    CHECK(feed_seg(3, 0) == LEP_FEED_SEGMENT);
    CHECK(lep_asm_feed(&A, d) == LEP_FEED_DISCARD);
    CHECK(feed_seg(4, 0) == LEP_FEED_FRAME);
    check_frame(0);
    CHECK(A.c.frames == 1);
    CHECK(A.c.discards == 7);
    CHECK(A.c.crc_errors == 0);
    CHECK(A.c.segments == 4);
}

static void test_invalid_segments_are_neutral(void)
{
    lep_asm_init(&A, true);
    CHECK(feed_seg(1, 1) == LEP_FEED_SEGMENT);
    CHECK(feed_seg(0, 9) == LEP_FEED_SEGMENT);   /* repeated frame segment, TTT = 0 */
    CHECK(feed_seg(2, 1) == LEP_FEED_SEGMENT);
    CHECK(feed_seg(3, 1) == LEP_FEED_SEGMENT);
    CHECK(feed_seg(0, 9) == LEP_FEED_SEGMENT);
    CHECK(feed_seg(4, 1) == LEP_FEED_FRAME);
    check_frame(1);
    CHECK(A.c.seg_invalid == 2);
}

static void test_out_of_order_and_missing(void)
{
    lep_asm_init(&A, true);
    /* Missing segment 2: segment 3 must not complete a frame. */
    CHECK(feed_seg(1, 2) == LEP_FEED_SEGMENT);
    CHECK(feed_seg(3, 2) == LEP_FEED_ERROR);
    CHECK(feed_seg(4, 2) == LEP_FEED_ERROR);   /* no frame in progress */
    CHECK(A.c.frames == 0);
    CHECK(A.c.frames_lost == 1);
    CHECK(A.c.seg_order_errors == 2);
    /* Recovery on the next segment 1. */
    CHECK(feed_seg(1, 3) == LEP_FEED_SEGMENT);
    CHECK(feed_seg(2, 3) == LEP_FEED_SEGMENT);
    CHECK(feed_seg(3, 3) == LEP_FEED_SEGMENT);
    CHECK(feed_seg(4, 3) == LEP_FEED_FRAME);
    check_frame(3);
    /* Starting mid-frame (segment 3 first) is ignored. */
    lep_asm_init(&A, true);
    CHECK(feed_seg(3, 4) == LEP_FEED_ERROR);
    CHECK(feed_seg(4, 4) == LEP_FEED_ERROR);
    CHECK(feed_seg(1, 4) == LEP_FEED_SEGMENT);
    /* Segment 1 again restarts the frame (the previous one is lost). */
    CHECK(feed_seg(1, 5) == LEP_FEED_SEGMENT);
    CHECK(A.c.frames_lost == 1);
    /* Invalid segment number 7. */
    CHECK(feed_seg(7, 5) == LEP_FEED_ERROR);
}

static void test_packet_errors_and_resync(void)
{
    lep_asm_init(&A, true);
    uint8_t p[LEP_PKT_BYTES];
    /* Corrupted packet mid-segment drops the segment. */
    for (int k = 0; k < 30; k++) { make_pkt(p, 1, k, 0); lep_asm_feed(&A, p); }
    make_pkt(p, 1, 30, 0);
    p[50] ^= 0x40;
    CHECK(lep_asm_feed(&A, p) == LEP_FEED_ERROR);
    CHECK(A.c.crc_errors == 1);
    make_pkt(p, 1, 31, 0);
    CHECK(lep_asm_feed(&A, p) == LEP_FEED_ERROR);  /* rest of segment is out of sync */
    /* Skipped packet number. */
    lep_asm_init(&A, true);
    for (int k = 0; k < 10; k++) { make_pkt(p, 1, k, 0); lep_asm_feed(&A, p); }
    make_pkt(p, 1, 11, 0);
    CHECK(lep_asm_feed(&A, p) == LEP_FEED_ERROR);
    CHECK(A.c.seq_errors == 1);
    /* Packet number >= 60 (telemetry rows / garbage). */
    make_pkt(p, 1, 61, 0);
    CHECK(lep_asm_feed(&A, p) == LEP_FEED_ERROR);
    /* A long run of garbage asks the driver for a resync. */
    lep_asm_init(&A, true);
    for (int i = 0; i < LEP_BAD_STREAK_RESYNC + 5; i++) {
        make_pkt(p, 1, 5 + (i % 50), 0);  /* never starts at packet 0 */
        lep_asm_feed(&A, p);
    }
    CHECK(A.need_resync);
    lep_asm_resynced(&A);
    CHECK(!A.need_resync);
    CHECK(A.c.resyncs == 1);
    CHECK(feed_seg(1, 6) == LEP_FEED_SEGMENT);
    CHECK(feed_seg(2, 6) == LEP_FEED_SEGMENT);
    lep_asm_resynced(&A);                       /* resync mid-frame loses it */
    CHECK(A.c.frames_lost == 1);
    CHECK(feed_seg(2, 6) == LEP_FEED_ERROR);
    /* CRC check disabled: corrupted data accepted. */
    lep_asm_init(&A, false);
    make_pkt(p, 1, 0, 0);
    p[3] ^= 0xFF;
    CHECK(lep_asm_feed(&A, p) == LEP_FEED_NONE);
}

static void test_crc_autodisable(void)
{
    lep_asm_init(&A, true);
    uint8_t p[LEP_PKT_BYTES];
    for (int i = 0; i < 2100; i++) {
        make_pkt(p, 1, i % 60, 0);
        p[2] ^= 0x55;  /* systematically "wrong" CRC */
        lep_asm_feed(&A, p);
    }
    CHECK(!A.check_crc);
}

static void test_tlinear(void)
{
    CHECK(lep_tlinear_to_dc(2732, LEP_RES_0_1K) == 1);    /* 273.2 K = 0.05 C -> 0.1 C (half away from zero) */
    CHECK(lep_tlinear_to_dc(2731, LEP_RES_0_1K) == -1);   /* 273.1 K = -0.05 C -> -0.1 C */
    CHECK(lep_tlinear_to_dc(3096, LEP_RES_0_1K) == 365);  /* 309.6 K = 36.45 -> 36.5 C */
    CHECK(lep_tlinear_to_dc(7231, LEP_RES_0_1K) == 4500); /* 723.1 K = 449.95 -> 450.0 C */
    CHECK(lep_tlinear_to_dc(2531, LEP_RES_0_1K) == -201); /* 253.1 K = -20.05 -> -20.1 C */
    CHECK(lep_tlinear_to_dc(30915, LEP_RES_0_01K) == 360);/* 309.15 K = 36.0 C */
    CHECK(lep_tlinear_to_dc(0, LEP_RES_0_1K) == -2732);
    CHECK(lep_tlinear_to_dc(65535, LEP_RES_0_1K) == 32767);  /* saturates */
    ps_thermal_frame_t f;
    lep_asm_init(&A, true);
    for (int i = 0; i < PS_THERM_PIXELS; i++) A.raw[i] = 2981;  /* 25.0 C (298.1 K = 24.95) */
    lep_asm_to_frame(&A, LEP_RES_0_1K, &f);
    CHECK(f.px[0] == 250 && f.px[PS_THERM_PIXELS - 1] == 250);
}

/* ---------- CCI with a fake camera ---------- */

typedef struct {
    uint16_t regs[0x30];
    uint32_t values[0x10000 >> 2];  /* by command base >> 2 ... sparse map below */
    int busy_polls;                 /* STATUS reads remaining busy */
    uint16_t log_cmds[64];
    int n_cmds;
    int fail_cmd;                   /* command base that returns an error */
    int writes;
} fake_t;

static fake_t F;

static uint32_t *fake_slot(uint16_t base)
{
    /* Hash command base into the value table (base is unique in the low 14 bits). */
    return &F.values[(base & 0x3FFF) >> 2];
}

static int f_read(void *ctx, uint16_t reg, uint16_t *v)
{
    (void)ctx;
    if (reg == LEP_REG_STATUS) {
        uint16_t st = F.regs[LEP_REG_STATUS / 2];
        if (F.busy_polls > 0) { F.busy_polls--; st |= LEP_STATUS_BUSY; }
        *v = st;
        return 0;
    }
    *v = F.regs[reg / 2];
    return 0;
}

static int f_write(void *ctx, uint16_t reg, uint16_t v)
{
    (void)ctx;
    F.writes++;
    F.regs[reg / 2] = v;
    if (reg == LEP_REG_COMMAND) {
        uint16_t base = v & ~3, type = v & 3;
        if (F.n_cmds < 64) F.log_cmds[F.n_cmds++] = v;
        F.busy_polls = 2;
        uint16_t st = LEP_STATUS_BOOT_MODE | LEP_STATUS_BOOT_STATUS;
        if (base == F.fail_cmd) st |= (uint16_t)((uint8_t)(-5) << 8);
        F.regs[LEP_REG_STATUS / 2] = st;
        if (base != F.fail_cmd) {
            uint32_t *slot = fake_slot(base);
            if (type == LEP_TYPE_SET)
                *slot = F.regs[LEP_REG_DATA0 / 2] | ((uint32_t)F.regs[LEP_REG_DATA0 / 2 + 1] << 16);
            else if (type == LEP_TYPE_GET) {
                F.regs[LEP_REG_DATA0 / 2] = (uint16_t)*slot;
                F.regs[LEP_REG_DATA0 / 2 + 1] = (uint16_t)(*slot >> 16);
            }
        }
    }
    return 0;
}

static void f_delay(void *ctx, uint32_t ms) { (void)ctx; (void)ms; }

static void test_cci(void)
{
    memset(&F, 0, sizeof(F));
    F.regs[LEP_REG_STATUS / 2] = LEP_STATUS_BOOT_MODE | LEP_STATUS_BOOT_STATUS;
    *fake_slot(LEP_CID_AGC_ENABLE) = 1;
    lep_cci_t c = { f_read, f_write, f_delay, NULL, 100 };
    CHECK(lep_cci_wait_boot(&c, 100) == LEP_CCI_OK);

    lep_setup_report_t rep;
    CHECK(lep_cci_setup_pyrosight(&c, &rep) == 0);
    CHECK(rep.agc_off && rep.radiometry_on && rep.tlinear_on && rep.low_gain && rep.res_0_1k &&
          rep.telemetry_off && rep.vsync_on);
    CHECK(*fake_slot(LEP_CID_AGC_ENABLE) == 0);
    CHECK(*fake_slot(LEP_CID_RAD_ENABLE) == 1);
    CHECK(*fake_slot(LEP_CID_RAD_TLINEAR_EN) == 1);
    CHECK(*fake_slot(LEP_CID_SYS_GAIN_MODE) == LEP_GAIN_LOW);
    CHECK(*fake_slot(LEP_CID_OEM_GPIO_MODE) == LEP_GPIO_MODE_VSYNC);
    /* Command words carry module, protection bit and type. */
    int saw_vsync_set = 0, saw_agc_set = 0;
    for (int i = 0; i < F.n_cmds; i++) {
        if (F.log_cmds[i] == 0x4855) saw_vsync_set = 1;  /* OEM GPIO mode SET */
        if (F.log_cmds[i] == 0x0101) saw_agc_set = 1;    /* AGC enable SET */
    }
    CHECK(saw_vsync_set && saw_agc_set);

    F.n_cmds = 0;
    CHECK(lep_cci_run_ffc(&c) == LEP_CCI_OK);
    CHECK(F.n_cmds == 1 && F.log_cmds[0] == 0x0242);

    /* Camera error result is reported and setup continues past it. */
    F.fail_cmd = LEP_CID_SYS_GAIN_MODE;
    CHECK(lep_cci_setup_pyrosight(&c, &rep) == -1);
    CHECK(!rep.low_gain && rep.vsync_on);
    CHECK(lep_cci_stats.last_result == 0 || lep_cci_stats.errors > 0);

    /* Busy forever -> timeout. */
    F.fail_cmd = 0;
    F.busy_polls = 1000000;
    CHECK(lep_cci_run_ffc(&c) == LEP_CCI_ERR_TIMEOUT);
    F.busy_polls = 0;

    /* FPA temperature: Kelvin x 100 -> deci-C. */
    *fake_slot(LEP_CID_SYS_FPA_TEMP_K) = 30315;  /* 30.0 C */
    int16_t dc = 0;
    CHECK(lep_cci_get_fpa_temp_dc(&c, &dc) == LEP_CCI_OK);
    CHECK(dc == 300);
}

int main(void)
{
    RUN(test_crc);
    RUN(test_full_frame_with_discards);
    RUN(test_invalid_segments_are_neutral);
    RUN(test_out_of_order_and_missing);
    RUN(test_packet_errors_and_resync);
    RUN(test_crc_autodisable);
    RUN(test_tlinear);
    RUN(test_cci);
    TEST_MAIN_END();
}
