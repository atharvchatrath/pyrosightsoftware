/*
 * Micro-OLED eyepiece display: MIPI-DSI DPI (video mode) via esp_lcd, two
 * frame buffers in PSRAM, the 320x240 RGB565 HUD frame (ps_fb_t) scaled by
 * an integer factor (3x -> 960x720) with the PPA and centred on the panel
 * (default 1024x768; black border). Brightness with DCS 0x51.
 * See panel_init_seq.h for the vendor init table to fill in.
 */
#ifndef MICROOLED_H
#define MICROOLED_H

#include <stdint.h>

#include "esp_err.h"
#include "pyrosight/ps_display.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int h_res, v_res;
    int hsync_pw, hbp, hfp;
    int vsync_pw, vbp, vfp;
    int dpi_clk_mhz;
    int lanes;
    int lane_mbps;
    int scale;              /* integer upscale of the 320x240 frame (3) */
    int pin_reset;          /* -1: none */
    int phy_ldo_chan;       /* ESP32-P4 LDO channel powering the DSI PHY (3 on the EV board) */
    int phy_ldo_mv;         /* 2500 */
    bool brightness_2bytes; /* DCS 0x51 with 2 parameter bytes */
    bool rgb888;            /* DSI pixel format: RGB888 (most micro-OLEDs) or RGB565 */
} microoled_config_t;

typedef struct {
    uint32_t frames, flip_timeouts, ppa_errors;
    uint32_t last_ppa_us, last_present_us;
} microoled_stats_t;

esp_err_t microoled_init(const microoled_config_t *cfg);

/* Scale fb into the back buffer and flip at the next vertical blank. */
esp_err_t microoled_present(const ps_fb_t *fb);

esp_err_t microoled_set_brightness(uint8_t level);

void microoled_get_stats(microoled_stats_t *out);

#ifdef __cplusplus
}
#endif

#endif /* MICROOLED_H */
