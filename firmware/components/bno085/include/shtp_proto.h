/*
 * Bosch/CEVA BNO085: SHTP transport framing and SH-2 report parsing.
 * Hardware independent (no ESP-IDF includes); host-tested in firmware/test_host.
 *
 * SHTP packet: 4-byte header
 *     [0] length LSB  [1] length MSB (bit 15 = continuation)  [2] channel  [3] sequence
 *   followed by (length - 4) cargo bytes. Length includes the header.
 * Channels: 0 SHTP command (advertisement), 1 executable (reset complete),
 *           2 sensor hub control, 3 input reports, 4 wake reports, 5 gyro RV.
 *
 * Input report batches (channel 3/4) start with a 0xFB base timestamp
 * reference (5 bytes, base delta in 100 us units, counted back from the
 * host interrupt), optionally 0xFA timestamp rebase, then reports of the form
 *     [id][seq][status: bits 1:0 accuracy, bits 7:2 delay MSB][delay LSB][data...]
 * with delay in 100 us units relative to the base timestamp.
 * Report lengths follow the SH-2 Reference Manual (1000-3625) report table.
 */
#ifndef SHTP_PROTO_H
#define SHTP_PROTO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BNO085_I2C_ADDR 0x4A   /* SA0 low; 0x4B with SA0 high */

#define SHTP_HDR_LEN 4
#define SHTP_MAX_PACKET 512    /* our receive buffer; longer cargo is skipped */

enum {
    SHTP_CHAN_COMMAND = 0,
    SHTP_CHAN_EXECUTABLE = 1,
    SHTP_CHAN_CONTROL = 2,
    SHTP_CHAN_REPORTS = 3,
    SHTP_CHAN_WAKE = 4,
    SHTP_CHAN_GYRO_RV = 5,
    SHTP_NUM_CHANNELS = 6,
};

/* SH-2 report ids used here. */
#define SH2_ACCELEROMETER       0x01
#define SH2_LINEAR_ACCELERATION 0x04
#define SH2_ROTATION_VECTOR     0x05
#define SH2_GAME_ROTATION_VECTOR 0x08
#define SH2_STEP_COUNTER        0x11
#define SH2_STEP_DETECTOR       0x18
#define SH2_CMD_RESPONSE        0xF1
#define SH2_COMMAND_REQUEST     0xF2
#define SH2_PRODUCT_ID_RESPONSE 0xF8
#define SH2_PRODUCT_ID_REQUEST  0xF9
#define SH2_TIMESTAMP_REBASE    0xFA
#define SH2_BASE_TIMESTAMP      0xFB
#define SH2_GET_FEATURE_RESPONSE 0xFC
#define SH2_SET_FEATURE_COMMAND 0xFD
#define SH2_GET_FEATURE_REQUEST 0xFE

#define SH2_EXEC_RESET_COMPLETE 0x01  /* channel 1, hub -> host */
#define SH2_EXEC_CMD_RESET      0x01  /* channel 1, host -> hub */

/* Q points (SH-2 reference manual, report descriptions). */
#define SH2_Q_GAME_RV    14
#define SH2_Q_LINEAR_ACC 8

typedef struct {
    uint16_t len;      /* total length including the header, 0 = no data */
    bool cont;
    uint8_t chan;
    uint8_t seq;
} shtp_hdr_t;

/* Returns false for an empty/invalid header (length 0 or 0xFFFF, bad channel). */
bool shtp_parse_header(const uint8_t *b, shtp_hdr_t *h);

/* Build header + payload; returns total length or 0 if it does not fit. */
size_t shtp_build_packet(uint8_t *out, size_t cap, uint8_t chan, uint8_t seq,
                         const uint8_t *payload, size_t n);

/* SH-2 Set Feature command (17 bytes). Returns 17. */
size_t sh2_build_set_feature(uint8_t out[17], uint8_t report_id, uint32_t interval_us,
                             uint32_t batch_us, uint16_t sensitivity, uint32_t specific);

/* Product ID request (2 bytes). */
size_t sh2_build_product_id_request(uint8_t out[2]);

/* Total length of an input/control report including its id, or -1 if unknown. */
int sh2_report_len(uint8_t id);

typedef enum {
    SH2_EV_GAME_RV = 1,
    SH2_EV_LINEAR_ACCEL,
    SH2_EV_STEP,
    SH2_EV_RESET_COMPLETE,  /* hub (re)booted: features must be enabled again */
    SH2_EV_ADVERTISEMENT,   /* channel 0 advertisement (also after a reset) */
    SH2_EV_PRODUCT_ID,
    SH2_EV_FEATURE_RESPONSE,
    SH2_EV_OTHER_REPORT,    /* known length, not used */
} sh2_event_type_t;

typedef struct {
    sh2_event_type_t type;
    uint8_t report_id;
    uint8_t accuracy;       /* status bits 1:0 */
    int32_t t_offset_us;    /* sample time relative to the host interrupt (<= 0 normally) */
    union {
        struct { float w, x, y, z; } q;          /* unit quaternion (real, i, j, k) */
        struct { float x, y, z; } v;             /* m/s^2 */
        struct { uint32_t latency_us; } step;
        struct { uint8_t report_id; uint8_t flags; uint32_t interval_us; } feature;
        struct { uint8_t reset_cause, sw_major, sw_minor; uint32_t part_no, build; uint16_t patch; } pid;
    } u;
} sh2_event_t;

typedef void (*sh2_event_cb_t)(void *ctx, const sh2_event_t *ev);

typedef struct {
    uint32_t packets;
    uint32_t reports;
    uint32_t unknown;       /* unknown report id: rest of the packet skipped */
    uint32_t truncated;     /* report longer than the remaining cargo */
    uint32_t resets;
    uint32_t seq_gaps;      /* SHTP sequence number jumps on the report channel */
    uint8_t last_seq[SHTP_NUM_CHANNELS];
    bool seq_valid[SHTP_NUM_CHANNELS];
} sh2_parser_t;

void sh2_parser_init(sh2_parser_t *p);

/*
 * Parse one SHTP packet's cargo (after the 4-byte header) received on `chan`
 * with sequence number `seq`. Calls cb for each event; returns the number of
 * events delivered.
 */
int sh2_parse_cargo(sh2_parser_t *p, uint8_t chan, uint8_t seq, const uint8_t *cargo, size_t n,
                    sh2_event_cb_t cb, void *ctx);

/* Convenience: parse a full packet (header + cargo); returns events or -1 on a bad header. */
int sh2_parse_packet(sh2_parser_t *p, const uint8_t *pkt, size_t n, sh2_event_cb_t cb, void *ctx);

/* |v| of a linear-acceleration event. */
float sh2_vec_mag(const sh2_event_t *ev);

#ifdef __cplusplus
}
#endif

#endif /* SHTP_PROTO_H */
