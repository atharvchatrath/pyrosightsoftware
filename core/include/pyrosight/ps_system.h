/*
 * PyroSight core: system integration.
 *
 * ps_system_t owns every subsystem's state and is the single object both the
 * firmware tasks and the host simulator drive. Data flow per thermal frame:
 *
 *   Lepton VoSPI ──► ps_system_on_frame()     denoise, display map, model input
 *                         │
 *         neural model ◄──┘ (firmware) or ps_system_run_classical()
 *                         │
 *                ps_system_on_detections()    distances, peaks, person alerts
 *
 *   BNO085 ──► ps_system_on_yaw / on_step / on_linear_accel
 *   buttons ─► ps_system_on_button
 *   ADC ─────► ps_system_on_battery
 *   timer ───► ps_system_tick()               health, nav confidence, alerts
 *   display ─► ps_system_render()             composited RGB565 frame
 *   audio ───► ps_system_next_alert()         phrase ids to play
 *
 * Not thread-safe by itself: the firmware serialises access with one mutex
 * (each call is short; the slow parts, VoSPI capture and inference, happen
 * outside the lock on private buffers).
 *
 * Memory: ~300 KB, so on the ESP32-P4 the instance lives in PSRAM.
 */
#ifndef PS_SYSTEM_H
#define PS_SYSTEM_H

#include "ps_alerts.h"
#include "ps_config.h"
#include "ps_detect.h"
#include "ps_display.h"
#include "ps_nav.h"
#include "ps_power.h"
#include "ps_thermal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PS_BTN_MARK_ENTRY = 0,  /* long press: "I am at the doorway" */
    PS_BTN_WHERE_OUT,       /* short press: speak the way out now */
    PS_BTN_PALETTE,         /* cycle white-hot / iron / night */
    PS_BTN_BRIGHTNESS,      /* cycle panel brightness */
} ps_button_t;

typedef struct {
    uint32_t frames;
    uint32_t frames_dropped;    /* frame_id gaps */
    uint32_t detections_runs;
    uint32_t last_pipeline_us;  /* filled by the caller with its own timer */
    uint32_t last_infer_us;
    uint32_t camera_losses;
    uint32_t imu_losses;
} ps_stats_t;

typedef struct {
    ps_config_t cfg;
    ps_display_settings_t disp;

    ps_thermal_t thermal;
    ps_thermal_frame_t frame;       /* latest denoised radiometric frame */
    ps_gray_frame_t display_gray;
    ps_gray_frame_t model_input;
    ps_dc_t scratch[PS_THERM_PIXELS];
    uint16_t labels[PS_THERM_PIXELS];

    ps_detections_t dets;
    bool have_frame;

    ps_nav_t nav;
    ps_alerts_t alerts;
    ps_power_t power;
    ps_power_policy_t policy;

    /* Health supervisor. */
    uint32_t t_last_frame_ms;
    uint32_t t_last_imu_ms;
    uint32_t t_last_dets_ms;
    bool camera_ok;
    bool imu_ok;
    bool imu_seen;

    ps_stats_t stats;
} ps_system_t;

/* cfg may be NULL for defaults. */
void ps_system_init(ps_system_t *s, const ps_config_t *cfg, uint32_t t_ms);

void ps_system_on_frame(ps_system_t *s, const ps_thermal_frame_t *raw);
void ps_system_run_classical(ps_system_t *s);
void ps_system_on_detections(ps_system_t *s, const ps_detections_t *dets, uint32_t t_ms);

void ps_system_on_yaw(ps_system_t *s, float yaw_rad, uint32_t t_ms);
void ps_system_on_step(ps_system_t *s, uint32_t t_ms);
void ps_system_on_linear_accel(ps_system_t *s, float mag, uint32_t t_ms);
void ps_system_on_button(ps_system_t *s, ps_button_t b, uint32_t t_ms);
void ps_system_on_battery(ps_system_t *s, uint16_t mv, uint32_t t_ms);

void ps_system_tick(ps_system_t *s, uint32_t t_ms);
void ps_system_render(ps_system_t *s, ps_fb_t *fb, uint32_t t_ms);
bool ps_system_next_alert(ps_system_t *s, ps_alert_t *out);

/* Panel brightness after the power-policy cap. */
uint8_t ps_system_brightness(const ps_system_t *s);

#ifdef __cplusplus
}
#endif

#endif /* PS_SYSTEM_H */
