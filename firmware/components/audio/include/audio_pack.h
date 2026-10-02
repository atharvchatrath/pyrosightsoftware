/*
 * "audio" flash partition format (written by tools/make_audio_clips.py) and
 * the fallback beep generator. Hardware independent; host-tested.
 *
 * Layout, all integers little endian:
 *   offset 0   header (24 bytes)
 *       char     magic[4]     "PSAU"
 *       uint16   version      1
 *       uint16   count        number of index entries (= PS_PHRASE_COUNT when built)
 *       uint32   sample_rate  16000
 *       uint8    bits         16
 *       uint8    channels     1
 *       uint16   reserved     0
 *       uint32   image_size   total bytes used in the partition
 *       uint32   index_crc    CRC-32 (zlib/ISO-HDLC) of the index entries
 *   offset 24  index: count x { uint32 offset; uint32 length; }  (bytes, from
 *              partition start; entry i is ps_phrase_t i; length 0 = missing)
 *   then       PCM data, signed 16-bit LE mono, each clip 4-byte aligned.
 */
#ifndef AUDIO_PACK_H
#define AUDIO_PACK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AUDIO_PACK_MAGIC "PSAU"
#define AUDIO_PACK_VERSION 1
#define AUDIO_PACK_HDR_BYTES 24
#define AUDIO_PACK_ENTRY_BYTES 8
#define AUDIO_PACK_MAX_ENTRIES 64
#define AUDIO_SAMPLE_RATE 16000

typedef enum {
    AUDIO_PACK_OK = 0,
    AUDIO_PACK_ERR_SHORT = -1,     /* buffer smaller than header + index */
    AUDIO_PACK_ERR_MAGIC = -2,     /* erased partition or not ours */
    AUDIO_PACK_ERR_VERSION = -3,
    AUDIO_PACK_ERR_FORMAT = -4,    /* rate/bits/channels unsupported */
    AUDIO_PACK_ERR_COUNT = -5,
    AUDIO_PACK_ERR_CRC = -6,
    AUDIO_PACK_ERR_BOUNDS = -7,    /* image larger than the partition */
} audio_pack_err_t;

typedef struct {
    uint32_t offset, length;
} audio_clip_ref_t;

typedef struct {
    uint16_t count;
    uint32_t sample_rate;
    uint32_t image_size;
    audio_clip_ref_t clip[AUDIO_PACK_MAX_ENTRIES];
    uint16_t n_valid;              /* entries with valid, in-bounds data */
} audio_pack_t;

/* Bytes needed to parse a pack with `count` entries (header + index). */
static inline size_t audio_pack_index_bytes(uint16_t count)
{
    return AUDIO_PACK_HDR_BYTES + (size_t)count * AUDIO_PACK_ENTRY_BYTES;
}

/* Read the entry count from a header (to know how much to read); 0 if bad magic. */
uint16_t audio_pack_peek_count(const uint8_t *hdr, size_t n);

/*
 * Parse header + index (buf holds at least audio_pack_index_bytes(count)).
 * part_size bounds every clip. Entries pointing outside the partition, with
 * odd length or overlapping the index are dropped (length set to 0) rather
 * than failing the whole pack.
 */
int audio_pack_parse(const uint8_t *buf, size_t n, size_t part_size, audio_pack_t *out);

/* Returns true and the clip location if phrase `id` has audio. */
bool audio_pack_clip(const audio_pack_t *p, unsigned id, uint32_t *offset, uint32_t *length);

uint32_t audio_crc32(const uint8_t *d, size_t n);

/* ---------- fallback beeps (when a clip is missing) ---------- */

typedef struct {
    uint16_t freq_hz;   /* 0 = silence */
    uint16_t ms;
} audio_tone_t;

#define AUDIO_BEEP_MAX_TONES 12

/*
 * Beep pattern for an alert priority (0 info .. 3 critical): more, faster and
 * higher beeps for higher priority so the wearer can tell them apart.
 * Returns the number of tones written.
 */
int audio_beep_pattern(int prio, audio_tone_t out[AUDIO_BEEP_MAX_TONES]);

/* Streaming tone renderer: fills buf with up to n samples; returns samples written (0 = done). */
typedef struct {
    const audio_tone_t *tones;
    int n_tones;
    int idx;
    uint32_t pos;         /* sample within the current tone */
    float phase;
    int16_t amplitude;
} audio_tone_gen_t;

void audio_tone_gen_init(audio_tone_gen_t *g, const audio_tone_t *tones, int n, int16_t amplitude);
size_t audio_tone_gen_fill(audio_tone_gen_t *g, int16_t *buf, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PACK_H */
