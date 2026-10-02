/* Host tests: audio partition index parser and beep generator.
 * Usage: test_audio_pack [audio.bin from tools/make_audio_clips.py] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../core/tests/ps_test.h"
#include "audio_pack.h"
#include "pyrosight/ps_alerts.h"

static void w16(uint8_t *b, unsigned v) { b[0] = (uint8_t)v; b[1] = (uint8_t)(v >> 8); }
static void w32(uint8_t *b, uint32_t v) { for (int i = 0; i < 4; i++) b[i] = (uint8_t)(v >> (8 * i)); }

/* Build a small image: 3 entries, entry 1 missing. */
static size_t build(uint8_t *img, size_t cap)
{
    memset(img, 0, cap);
    const uint16_t count = 3;
    size_t idx_end = audio_pack_index_bytes(count);
    memcpy(img, "PSAU", 4);
    w16(img + 4, 1); w16(img + 6, count); w32(img + 8, 16000); img[12] = 16; img[13] = 1;
    uint32_t off0 = (uint32_t)idx_end, len0 = 200, off2 = off0 + 200, len2 = 64;
    uint8_t *e = img + AUDIO_PACK_HDR_BYTES;
    w32(e, off0); w32(e + 4, len0);
    w32(e + 8, 0); w32(e + 12, 0);
    w32(e + 16, off2); w32(e + 20, len2);
    size_t size = off2 + len2;
    w32(img + 16, (uint32_t)size);
    w32(img + 20, audio_crc32(e, count * AUDIO_PACK_ENTRY_BYTES));
    return size;
}

static void test_parse(void)
{
    uint8_t img[1024];
    size_t size = build(img, sizeof(img));
    audio_pack_t p;
    CHECK(audio_pack_peek_count(img, 24) == 3);
    CHECK(audio_pack_parse(img, size, 4096, &p) == AUDIO_PACK_OK);
    CHECK(p.count == 3 && p.n_valid == 2);
    uint32_t off, len;
    CHECK(audio_pack_clip(&p, 0, &off, &len) && off == 48 && len == 200);
    CHECK(!audio_pack_clip(&p, 1, &off, &len));
    CHECK(audio_pack_clip(&p, 2, &off, &len) && len == 64);
    CHECK(!audio_pack_clip(&p, 3, &off, &len));
    CHECK(!audio_pack_clip(&p, 1000, NULL, NULL));

    /* Errors. */
    CHECK(audio_pack_parse(img, 10, 4096, &p) == AUDIO_PACK_ERR_SHORT);
    CHECK(audio_pack_parse(img, 30, 4096, &p) == AUDIO_PACK_ERR_SHORT);   /* index cut */
    CHECK(audio_pack_parse(img, size, 100, &p) == AUDIO_PACK_ERR_BOUNDS);  /* partition too small */
    uint8_t erased[64];
    memset(erased, 0xFF, sizeof(erased));
    CHECK(audio_pack_parse(erased, sizeof(erased), 4096, &p) == AUDIO_PACK_ERR_MAGIC);
    CHECK(audio_pack_peek_count(erased, sizeof(erased)) == 0);
    uint8_t bad[1024];
    memcpy(bad, img, sizeof(bad)); bad[4] = 2;
    CHECK(audio_pack_parse(bad, size, 4096, &p) == AUDIO_PACK_ERR_VERSION);
    memcpy(bad, img, sizeof(bad)); w32(bad + 8, 22050);
    CHECK(audio_pack_parse(bad, size, 4096, &p) == AUDIO_PACK_ERR_FORMAT);
    memcpy(bad, img, sizeof(bad)); bad[30] ^= 1;
    CHECK(audio_pack_parse(bad, size, 4096, &p) == AUDIO_PACK_ERR_CRC);
    memcpy(bad, img, sizeof(bad)); w16(bad + 6, 0);
    CHECK(audio_pack_parse(bad, size, 4096, &p) == AUDIO_PACK_ERR_COUNT);

    /* A single bad entry (beyond image / odd length) is dropped, not fatal. */
    memcpy(bad, img, sizeof(bad));
    uint8_t *e = bad + AUDIO_PACK_HDR_BYTES;
    w32(e + 20, 65);                                   /* odd length */
    w32(e + 4, 100000);                                /* out of bounds */
    w32(bad + 20, audio_crc32(e, 3 * AUDIO_PACK_ENTRY_BYTES));
    CHECK(audio_pack_parse(bad, size, 4096, &p) == AUDIO_PACK_OK);
    CHECK(p.n_valid == 0);
    /* Clip overlapping the index is rejected. */
    memcpy(bad, img, sizeof(bad));
    w32(e, 8);
    w32(bad + 20, audio_crc32(e, 3 * AUDIO_PACK_ENTRY_BYTES));
    CHECK(audio_pack_parse(bad, size, 4096, &p) == AUDIO_PACK_OK);
    CHECK(!audio_pack_clip(&p, 0, NULL, NULL) && audio_pack_clip(&p, 2, NULL, NULL));
}

static void test_beeps(void)
{
    audio_tone_t t[AUDIO_BEEP_MAX_TONES];
    int prev_ms = 0;
    for (int prio = 0; prio <= 3; prio++) {
        int n = audio_beep_pattern(prio, t);
        CHECK(n > 0 && n <= AUDIO_BEEP_MAX_TONES);
        int ms = 0, beeps = 0;
        for (int i = 0; i < n; i++) { ms += t[i].ms; if (t[i].freq_hz) beeps++; }
        CHECK(ms > prev_ms);   /* more urgent = longer pattern */
        CHECK(beeps >= prio + 1);
        prev_ms = ms;

        audio_tone_gen_t g;
        audio_tone_gen_init(&g, t, n, 8000);
        int16_t buf[256];
        size_t total = 0, got;
        int peak = 0;
        while ((got = audio_tone_gen_fill(&g, buf, 256)) > 0) {
            for (size_t i = 0; i < got; i++) if (abs(buf[i]) > peak) peak = abs(buf[i]);
            total += got;
        }
        CHECK(total == (size_t)ms * 16);
        CHECK(peak > 7000 && peak <= 8000);
    }
}

static void test_image_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL);
    if (!f) return;
    static uint8_t img[4 << 20];
    size_t n = fread(img, 1, sizeof(img), f);
    fclose(f);
    audio_pack_t p;
    CHECK(audio_pack_parse(img, n, 0x200000, &p) == AUDIO_PACK_OK);
    CHECK(p.count == PS_PHRASE_COUNT);
    CHECK(!audio_pack_clip(&p, PS_PHRASE_NONE, NULL, NULL));
    CHECK(p.n_valid == PS_PHRASE_COUNT - 1);
    for (unsigned i = 1; i < PS_PHRASE_COUNT; i++) {
        uint32_t off, len;
        CHECK(audio_pack_clip(&p, i, &off, &len) && (off & 3) == 0 && off + len <= n);
    }
    printf("parsed %s: %u clips, %u bytes\n", path, p.n_valid, (unsigned)p.image_size);
}

int main(int argc, char **argv)
{
    RUN(test_parse);
    RUN(test_beeps);
    if (argc > 1) test_image_file(argv[1]);
    TEST_MAIN_END();
}
