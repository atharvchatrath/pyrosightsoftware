/*
 * PyroSight core: eyepiece frame compositor.
 *
 * Renders thermal image + detection boxes + navigation arrow + status into an
 * RGB565 framebuffer. The firmware hands the buffer to the micro-OLED driver,
 * which scales it to panel resolution with the ESP32-P4 PPA (2D accelerator).
 *
 * Readability in the dark/smoke:
 *   - every HUD glyph and line is drawn with a black outline, so it reads on
 *     a white-hot background and on black alike;
 *   - NIGHT palette is amber monochrome at reduced luminance to protect dark
 *     adaptation (the eye is millimetres from an OLED);
 *   - brightness is a separate panel setting (ps_display_settings_t).
 *
 * Boxes: people white, fire purple, the exit green. The exit box is drawn
 * where the device believes the doorway is, only when the door is the next
 * point on the way out (so it never points through a wall it already knows
 * about) and the navigation estimate is still trusted.
 */
#ifndef PS_DISPLAY_H
#define PS_DISPLAY_H

#include "ps_alerts.h"
#include "ps_config.h"
#include "ps_detect.h"
#include "ps_nav.h"
#include "ps_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PS_DISP_SCALE 2
#define PS_DISP_W (PS_THERM_W * PS_DISP_SCALE)  /* 320 */
#define PS_DISP_H (PS_THERM_H * PS_DISP_SCALE)  /* 240 */

typedef struct {
    uint16_t px[PS_DISP_W * PS_DISP_H]; /* RGB565, row-major */
} ps_fb_t;

#define PS_RGB565(r, g, b) ((uint16_t)((((r) & 0xF8) << 8) | (((g) & 0xFC) << 3) | ((b) >> 3)))

/* Box colours: one meaning per colour, consistent everywhere. */
#define PS_COLOR_PERSON PS_RGB565(255, 255, 255) /* white  */
#define PS_COLOR_FIRE   PS_RGB565(200, 80, 255)  /* purple */
#define PS_COLOR_EXIT   PS_RGB565(40, 255, 80)   /* green  */

/* Physical size used to draw the exit marker in the camera view. */
#define PS_EXIT_W_M 0.9f
#define PS_EXIT_H_M 2.0f

typedef enum {
    PS_PALETTE_WHITE_HOT = 0,
    PS_PALETTE_IRON,
    PS_PALETTE_NIGHT,
    PS_PALETTE_COUNT,
} ps_palette_t;

typedef struct {
    ps_palette_t palette;
    uint8_t brightness;      /* 0..255, applied by the panel driver */
    bool fire_overlay;       /* tint pixels above the fire threshold */
    bool show_home_bearing;  /* also mark straight-line bearing to the door */
} ps_display_settings_t;

typedef struct {
    const ps_gray_frame_t *thermal;          /* NULL when the camera is lost */
    const ps_thermal_frame_t *radiometric;   /* for fire overlay; may be NULL */
    const ps_detections_t *dets;             /* may be NULL */
    const ps_nav_guidance_t *nav;            /* may be NULL */
    ps_navconf_level_t nav_level;
    bool nav_tracking;                       /* entry has been marked */
    bool camera_ok;
    bool imu_ok;
    ps_detector_kind_t detector;
    uint8_t battery_pct;
    bool battery_low;
    uint32_t t_ms;                           /* drives blinking */
} ps_hud_t;

void ps_display_settings_default(ps_display_settings_t *s);

void ps_display_render(ps_fb_t *fb, const ps_display_settings_t *s,
                       const ps_config_t *cfg, const ps_hud_t *hud);

/* Drawing primitives (exposed for the boot screen and tests). */
uint16_t ps_rgb565(uint8_t r, uint8_t g, uint8_t b);
void ps_fb_fill(ps_fb_t *fb, uint16_t c);
void ps_fb_rect(ps_fb_t *fb, int x, int y, int w, int h, int thick, uint16_t c);
void ps_fb_fill_rect(ps_fb_t *fb, int x, int y, int w, int h, uint16_t c);
void ps_fb_line(ps_fb_t *fb, int x0, int y0, int x1, int y1, uint16_t c);
void ps_fb_triangle(ps_fb_t *fb, int x0, int y0, int x1, int y1, int x2, int y2, uint16_t c);
void ps_fb_circle(ps_fb_t *fb, int cx, int cy, int r, uint16_t c);
/* Text: 5x7 font, upper-case ASCII; `outline` draws a black halo. */
int ps_fb_text(ps_fb_t *fb, int x, int y, const char *s, int scale, uint16_t c, bool outline);
int ps_fb_text_width(const char *s, int scale);

#ifdef __cplusplus
}
#endif

#endif /* PS_DISPLAY_H */
