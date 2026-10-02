/*
 * Audio output: ES8311 codec on the ESP32-P4-Function-EV-Board, I2S at
 * 16 kHz / 16 bit, 3.5 mm jack. Plays ps_alert_t phrase sequences from the
 * "audio" partition (audio_pack.h); a missing clip (or a missing/erased
 * partition) falls back to a priority-coded beep pattern, so an alert is
 * never silent.
 */
#ifndef AUDIO_H
#define AUDIO_H

#include <stdbool.h>
#include <stdint.h>

#include "driver/i2c_master.h"
#include "esp_err.h"
#include "pyrosight/ps_alerts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    i2c_master_bus_handle_t i2c_bus;   /* bus the ES8311 is on */
    uint8_t codec_addr_7bit;           /* 0x18 on the EV board */
    int i2s_port;
    int pin_mclk, pin_bclk, pin_ws, pin_dout;
    int pin_pa;                        /* speaker amplifier enable, -1: none */
    uint8_t volume;                    /* 0..100 */
    const char *partition_label;       /* "audio" */
} audio_config_t;

typedef struct {
    uint32_t alerts, clips, beeps, missing_clips, aborted, write_errors;
    bool pack_ok;
    int pack_err;                      /* audio_pack_err_t when !pack_ok */
    uint16_t clips_available;
} audio_stats_t;

/* Return true to stop playback early (a more urgent alert is waiting). */
typedef bool (*audio_abort_fn)(void *ctx, int playing_prio);

esp_err_t audio_init(const audio_config_t *cfg);

/* Blocking: plays every part with a short gap, then returns. */
esp_err_t audio_play_alert(const ps_alert_t *a, audio_abort_fn should_abort, void *ctx);

/* Beep pattern for a priority (also used at boot as a self-test). */
esp_err_t audio_beep(int prio);

esp_err_t audio_set_volume(uint8_t volume);
void audio_get_stats(audio_stats_t *out);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_H */
