/*
 * FLIR Lepton 3.x VoSPI protocol: packet CRC, segment/frame assembly and
 * TLinear conversion. Hardware independent (no ESP-IDF includes) so it is
 * unit-tested on the host (firmware/test_host/) with synthetic packets.
 *
 * VoSPI recap (Lepton Engineering Datasheet, "VoSPI" chapter):
 *   - packet = 164 bytes: ID (2, big endian), CRC (2, big endian), payload (160)
 *   - ID bits 11:0 = packet number 0..59; bits 15:12 = TTT segment number,
 *     valid only in packet 20 (Lepton 3.x); 0 there means "invalid segment,
 *     discard" (Lepton 3.x emits repeated frames as invalid segments).
 *   - discard packets: ID & 0x0F00 == 0x0F00 (xFxx); they keep the bus clocked
 *     while no data is ready and must be ignored.
 *   - with telemetry disabled a segment is 60 packets = 30 rows x 160 px
 *     (2 packets per row, 80 pixels of 16 bit big-endian per packet);
 *     a frame is 4 segments (rows 0-29, 30-59, 60-89, 90-119).
 *   - CRC: CRC-16 CCITT (x^16 + x^12 + x^5 + 1), seed 0, over the whole packet
 *     with the 4 MSBs of the ID and the 16 CRC bits set to zero.
 *     (This is CRC-16/XMODEM; see lep_crc16(). If a sensor revision ever
 *     disagrees, the assembler auto-disables the check, see crc_autodisable.)
 *   - losing sync: idle CS (no clocks) for > 185 ms, then restart reading.
 */
#ifndef LEPTON_VOSPI_PROTO_H
#define LEPTON_VOSPI_PROTO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "pyrosight/ps_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LEP_PKT_BYTES      164
#define LEP_PKT_PAYLOAD    160
#define LEP_PKTS_PER_SEG   60
#define LEP_SEGS_PER_FRAME 4
#define LEP_ROWS_PER_SEG   30
#define LEP_SEG_PKT_TTT    20      /* packet carrying the segment number */
#define LEP_RESYNC_IDLE_MS 200     /* > 185 ms per datasheet */

/* Error streak (bad packets without a good segment) that triggers a resync. */
#define LEP_BAD_STREAK_RESYNC 240

typedef enum {
    LEP_RES_0_1K = 0,   /* TLinear in Kelvin x 10  (needed for low gain / fire) */
    LEP_RES_0_01K = 1,  /* TLinear in Kelvin x 100 (max ~382 C: high gain only) */
} lep_tlinear_res_t;

typedef struct {
    uint32_t packets;        /* non-discard packets processed */
    uint32_t discards;
    uint32_t crc_errors;
    uint32_t seq_errors;     /* unexpected packet number */
    uint32_t seg_invalid;    /* segment number 0 (repeated frame) */
    uint32_t seg_order_errors; /* segment out of 1..4 order */
    uint32_t segments;       /* valid segments committed */
    uint32_t frames;         /* complete frames */
    uint32_t frames_lost;    /* frames abandoned part way */
    uint32_t resyncs;
} lep_vospi_counters_t;

typedef enum {
    LEP_FEED_NONE = 0,       /* packet consumed, nothing completed */
    LEP_FEED_DISCARD,        /* discard packet */
    LEP_FEED_ERROR,          /* CRC / sequence error, segment dropped */
    LEP_FEED_SEGMENT,        /* a segment completed (valid or invalid) */
    LEP_FEED_FRAME,          /* segment 4 completed: frame ready in raw[] */
} lep_feed_t;

typedef struct {
    /* Assembled frame, raw 16-bit sensor values (TLinear units). */
    uint16_t raw[PS_THERM_PIXELS];
    /* Segment being received; committed to raw[] after packet 59. */
    uint16_t seg[LEP_PKTS_PER_SEG * (LEP_PKT_PAYLOAD / 2)];

    bool check_crc;
    bool crc_autodisable;    /* turn the CRC check off if no packet ever passes */
    bool in_segment;         /* received packet 0 of the current segment */
    int expected_pkt;
    int seg_no;              /* TTT from packet 20, -1 before it */
    int expected_seg;        /* 1..4 */
    bool frame_in_progress;
    uint32_t bad_streak;
    bool need_resync;        /* the driver should idle CS > 185 ms and call lep_asm_resynced() */
    uint32_t crc_ok_ever;

    lep_vospi_counters_t c;
} lep_asm_t;

void lep_asm_init(lep_asm_t *a, bool check_crc);

/* Feed one 164-byte packet. */
lep_feed_t lep_asm_feed(lep_asm_t *a, const uint8_t *pkt);

/* The driver idled CS; restart from packet 0 / segment 1. */
void lep_asm_resynced(lep_asm_t *a);

/* CRC-16 CCITT (poly 0x1021, seed 0, no reflection, no xorout). */
uint16_t lep_crc16(const uint8_t *data, size_t n);
uint16_t lep_crc16_update(uint16_t crc, uint8_t byte);

/* Packet helpers. */
static inline uint16_t lep_pkt_id(const uint8_t *p) { return (uint16_t)((p[0] << 8) | p[1]); }
static inline bool lep_pkt_is_discard(const uint8_t *p) { return (p[0] & 0x0F) == 0x0F; }
static inline int lep_pkt_number(const uint8_t *p) { return lep_pkt_id(p) & 0x0FFF; }
static inline int lep_pkt_ttt(const uint8_t *p) { return (p[0] >> 4) & 0x7; }
bool lep_pkt_crc_ok(const uint8_t *pkt);
/* Fill in the CRC field of a packet (used by tests and the simulator). */
void lep_pkt_set_crc(uint8_t *pkt);

/* TLinear sensor value -> deci-Celsius, rounded, saturated to int16. */
ps_dc_t lep_tlinear_to_dc(uint16_t raw, lep_tlinear_res_t res);

/* Convert the assembled raw[] into a core frame (frame_id/t_ms set by caller). */
void lep_asm_to_frame(const lep_asm_t *a, lep_tlinear_res_t res, ps_thermal_frame_t *out);

#ifdef __cplusplus
}
#endif

#endif /* LEPTON_VOSPI_PROTO_H */
