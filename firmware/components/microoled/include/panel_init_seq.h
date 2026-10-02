/*
 * ============================================================================
 *  VENDOR PANEL INITIALISATION SEQUENCE  --  FILL IN FROM THE PANEL DATASHEET
 * ============================================================================
 *
 * Micro-OLED microdisplays (0.39" 1024x768 class, e.g. Sony ECX-series,
 * SeeYA, BOE) each need a vendor-specific register sequence before they
 * accept MIPI-DSI video: power sequencing, gamma, MIPI lane/format setup,
 * scan direction, luminance mode. That sequence is in the panel's datasheet or
 * the vendor's reference driver and is NOT generic; nothing here is from a
 * real panel. Only the standard MIPI DCS commands below are universal.
 *
 * Each entry: DCS/generic command byte, parameter bytes, parameter count,
 * delay after sending (ms). Sent over DSI in command (LP) mode with
 * esp_lcd_panel_io_tx_param() BEFORE the DPI video stream starts
 * (panel_init_pre_video), and AFTER it started (panel_init_post_video) for
 * panels that must see video before display-on.
 *
 * Also set in menuconfig (PyroSight -> Micro-OLED): resolution, porches,
 * pixel clock, lane count and bit rate, all from the same datasheet.
 */
#ifndef PANEL_INIT_SEQ_H
#define PANEL_INIT_SEQ_H

#include <stdint.h>

typedef struct {
    uint8_t cmd;
    uint8_t data[16];
    uint8_t len;
    uint16_t delay_ms;
} panel_init_cmd_t;

/* Standard MIPI DCS command codes. */
#define DCS_SOFT_RESET        0x01
#define DCS_EXIT_SLEEP_MODE   0x11
#define DCS_SET_DISPLAY_OFF   0x28
#define DCS_SET_DISPLAY_ON    0x29
#define DCS_SET_PIXEL_FORMAT  0x3A   /* 0x55 = 16 bpp, 0x66 = 18, 0x77 = 24 */
#define DCS_SET_BRIGHTNESS    0x51   /* 1 byte (some panels: 2 bytes, MSB first) */
#define DCS_WRITE_CTRL_DISPLAY 0x53  /* bit 5 BCTRL: brightness control on */

static const panel_init_cmd_t panel_init_pre_video[] = {
    { DCS_SOFT_RESET, { 0 }, 0, 20 },
    /* ---- VENDOR SEQUENCE START: replace with the datasheet's table ---- */
    /* { 0xFE, { 0x01 }, 1, 0 },   e.g. page select                         */
    /* { 0x..., { ... }, n, 0 },   power / gamma / MIPI setup               */
    /* ---- VENDOR SEQUENCE END ------------------------------------------ */
    /* DCS_SET_PIXEL_FORMAT is sent by microoled.c to match the configured
     * DSI pixel format (RGB565 -> 0x55, RGB888 -> 0x77). */
    { DCS_WRITE_CTRL_DISPLAY, { 0x20 }, 1, 0 },        /* enable DCS brightness control */
    { DCS_SET_BRIGHTNESS, { 0x80 }, 1, 0 },
    { DCS_EXIT_SLEEP_MODE, { 0 }, 0, 120 },            /* datasheets typically require 120 ms */
};

static const panel_init_cmd_t panel_init_post_video[] = {
    { DCS_SET_DISPLAY_ON, { 0 }, 0, 20 },
};

#endif /* PANEL_INIT_SEQ_H */
