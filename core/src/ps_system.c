#include "pyrosight/ps_system.h"

#include <string.h>

void ps_system_init(ps_system_t *s, const ps_config_t *cfg, uint32_t t_ms)
{
    memset(s, 0, sizeof(*s));
    if (cfg) s->cfg = *cfg;
    else ps_config_default(&s->cfg);
    ps_display_settings_default(&s->disp);
    ps_thermal_init(&s->thermal);
    ps_nav_init(&s->nav);
    ps_alerts_init(&s->alerts);
    ps_power_init(&s->power);
    ps_power_policy(&s->power, &s->policy);
    s->t_last_frame_ms = s->t_last_imu_ms = s->t_last_dets_ms = t_ms;
    s->camera_ok = false; /* until the first frame arrives */
    s->imu_ok = false;
}

void ps_system_on_frame(ps_system_t *s, const ps_thermal_frame_t *raw)
{
    if (s->have_frame && raw->frame_id != s->frame.frame_id + 1)
        s->stats.frames_dropped += raw->frame_id - s->frame.frame_id - 1;
    s->frame = *raw;
    ps_thermal_denoise(&s->thermal, &s->cfg, &s->frame, s->scratch);
    ps_thermal_to_display(&s->thermal, &s->cfg, &s->frame, &s->display_gray);
    ps_thermal_to_model_input(&s->frame, &s->model_input);
    s->have_frame = true;
    s->t_last_frame_ms = raw->t_ms;
    s->stats.frames++;
    if (!s->camera_ok) {
        if (s->stats.frames > 1)
            ps_alerts_push(&s->alerts, PS_PRIO_WARNING, raw->t_ms, PS_PHRASE_CAMERA_RESTORED, PS_PHRASE_NONE, PS_PHRASE_NONE);
        s->camera_ok = true;
    }
}

void ps_system_run_classical(ps_system_t *s)
{
    if (!s->have_frame) return;
    ps_detections_t d;
    ps_detect_classical(&s->cfg, &s->frame, s->labels, &d);
    ps_system_on_detections(s, &d, s->frame.t_ms);
}

void ps_system_on_detections(ps_system_t *s, const ps_detections_t *dets, uint32_t t_ms)
{
    s->dets = *dets;
    /* Drop low-confidence boxes before they reach the wearer. */
    uint8_t w = 0;
    for (uint8_t i = 0; i < s->dets.n; i++)
        if (s->dets.d[i].score >= s->cfg.score_threshold) s->dets.d[w++] = s->dets.d[i];
    s->dets.n = w;
    if (s->have_frame) ps_detect_fill_peaks(&s->frame, &s->dets);
    ps_estimate_distances(&s->cfg, &s->dets);
    s->t_last_dets_ms = t_ms;
    s->stats.detections_runs++;
    ps_alerts_update_detections(&s->alerts, &s->cfg, &s->dets, t_ms);
}

static void imu_alive(ps_system_t *s, uint32_t t_ms)
{
    s->t_last_imu_ms = t_ms;
    s->imu_seen = true;
    s->imu_ok = true;
}

void ps_system_on_yaw(ps_system_t *s, float yaw_rad, uint32_t t_ms)
{
    imu_alive(s, t_ms);
    ps_nav_on_yaw(&s->nav, &s->cfg, yaw_rad, t_ms);
}

void ps_system_on_step(ps_system_t *s, uint32_t t_ms)
{
    imu_alive(s, t_ms);
    ps_nav_on_step(&s->nav, &s->cfg, t_ms);
}

void ps_system_on_linear_accel(ps_system_t *s, float mag, uint32_t t_ms)
{
    imu_alive(s, t_ms);
    ps_nav_on_linear_accel(&s->nav, &s->cfg, mag, t_ms);
}

void ps_system_on_button(ps_system_t *s, ps_button_t b, uint32_t t_ms)
{
    ps_nav_guidance_t g;
    switch (b) {
    case PS_BTN_MARK_ENTRY:
        ps_nav_mark_entry(&s->nav, t_ms);
        ps_alerts_init(&s->alerts);
        ps_alerts_push(&s->alerts, PS_PRIO_INFO, t_ms, PS_PHRASE_ENTRY_MARKED, PS_PHRASE_NONE, PS_PHRASE_NONE);
        break;
    case PS_BTN_WHERE_OUT:
        ps_nav_guidance(&s->nav, &g);
        ps_alerts_request_direction(&s->alerts, &g, t_ms);
        break;
    case PS_BTN_PALETTE:
        s->disp.palette = (ps_palette_t)((s->disp.palette + 1) % PS_PALETTE_COUNT);
        break;
    case PS_BTN_BRIGHTNESS:
        s->disp.brightness = s->disp.brightness >= 224 ? 32 : (uint8_t)(s->disp.brightness + 64);
        break;
    }
}

void ps_system_on_battery(ps_system_t *s, uint16_t mv, uint32_t t_ms)
{
    if (ps_power_update(&s->power, &s->cfg, mv)) {
        if (s->power.mode == PS_PWR_ECO)
            ps_alerts_push(&s->alerts, PS_PRIO_WARNING, t_ms, PS_PHRASE_BATTERY_LOW, PS_PHRASE_NONE, PS_PHRASE_NONE);
        else if (s->power.mode == PS_PWR_CRITICAL)
            ps_alerts_push(&s->alerts, PS_PRIO_CRITICAL, t_ms, PS_PHRASE_BATTERY_CRITICAL, PS_PHRASE_NONE, PS_PHRASE_NONE);
    }
    ps_power_policy(&s->power, &s->policy);
}

void ps_system_tick(ps_system_t *s, uint32_t t_ms)
{
    /* Health supervisor: sensors that stop talking are announced once. */
    if (s->camera_ok && t_ms - s->t_last_frame_ms > s->cfg.thermal_timeout_ms) {
        s->camera_ok = false;
        s->stats.camera_losses++;
        ps_alerts_push(&s->alerts, PS_PRIO_CRITICAL, t_ms, PS_PHRASE_CAMERA_LOST, PS_PHRASE_NONE, PS_PHRASE_NONE);
    }
    if (s->imu_ok && s->imu_seen && t_ms - s->t_last_imu_ms > s->cfg.imu_timeout_ms) {
        s->imu_ok = false;
        s->stats.imu_losses++;
        ps_alerts_push(&s->alerts, PS_PRIO_WARNING, t_ms, PS_PHRASE_MOTION_LOST, PS_PHRASE_NONE, PS_PHRASE_NONE);
    }
    /* Stale detections are worse than none: clear boxes that stopped updating. */
    if (s->dets.n && t_ms - s->t_last_dets_ms > s->cfg.detector_timeout_ms) s->dets.n = 0;

    ps_nav_tick(&s->nav, &s->cfg, t_ms);
    ps_nav_guidance_t g;
    ps_nav_guidance(&s->nav, &g);
    ps_alerts_update_nav(&s->alerts, &s->cfg, &g, t_ms);
}

void ps_system_render(ps_system_t *s, ps_fb_t *fb, uint32_t t_ms)
{
    ps_nav_guidance_t g;
    ps_nav_guidance(&s->nav, &g);
    ps_hud_t h = {
        .thermal = s->have_frame ? &s->display_gray : NULL,
        .radiometric = s->have_frame ? &s->frame : NULL,
        .dets = &s->dets,
        .nav = &g,
        .nav_level = s->alerts.level,
        .nav_tracking = s->nav.state != PS_NAV_IDLE,
        .camera_ok = s->camera_ok,
        .imu_ok = s->imu_ok || !s->imu_seen,
        .detector = s->dets.source,
        .battery_pct = s->power.pct,
        .battery_low = s->power.mode != PS_PWR_NORMAL,
        .t_ms = t_ms,
    };
    ps_display_render(fb, &s->disp, &s->cfg, &h);
}

bool ps_system_next_alert(ps_system_t *s, ps_alert_t *out)
{
    return ps_alerts_pop(&s->alerts, out);
}

uint8_t ps_system_brightness(const ps_system_t *s)
{
    return s->disp.brightness < s->policy.max_brightness ? s->disp.brightness : s->policy.max_brightness;
}
