/*
 * PyroSight core: tunable parameters.
 *
 * Every threshold lives here so field calibration touches one struct.
 * ps_config_default() returns the values used by the firmware and simulator.
 */
#ifndef PS_CONFIG_H
#define PS_CONFIG_H

#include "ps_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* ---- Thermal pipeline ---- */
    uint8_t temporal_shift;      /* IIR strength: new = old + (in-old) >> shift. 0 disables. */
    ps_dc_t motion_threshold_dc; /* pixel change that bypasses the IIR (prevents ghosting) */
    bool median3x3;              /* spatial 3x3 median for salt-and-pepper noise */
    ps_dc_t min_display_span_dc; /* never stretch less than this (avoids noise blow-up) */
    uint8_t lo_percentile;       /* display window percentiles */
    uint8_t hi_percentile;
    uint8_t detail_gain_q4;      /* unsharp-mask gain in 1/16 units, 0 disables */

    /* ---- Classical detector (fallback when no NN model is loaded) ---- */
    ps_dc_t fire_min_dc;   /* pixels at or above this are flame/fire candidates */
    ps_dc_t person_min_dc; /* human surface temperature band (clothing/skin) */
    ps_dc_t person_max_dc;
    ps_dc_t person_contrast_dc; /* min difference from the row background */
    uint16_t fire_min_area;
    uint16_t person_min_area;
    float nms_iou;
    float score_threshold;

    /* ---- Distance estimation ---- */
    float hfov_deg;         /* Lepton 3.5: 57 degrees horizontal */
    float person_height_m;  /* prior for the long axis of a person's box */

    /* ---- Navigation ---- */
    float step_length_m;        /* calibrated per user (tools/calibrate_steps.py) */
    float turn_threshold_deg;   /* yaw change that counts as a turn */
    float turn_window_ms;       /* ... within this window */
    float breadcrumb_spacing_m; /* also drop a breadcrumb every N metres */
    float waypoint_reached_m;   /* pop a return waypoint when this close */
    bool  heading_snap;         /* heuristic drift elimination to building axes */
    float heading_snap_deg;     /* within this many degrees of an axis, nudge */
    float heading_snap_gain;    /* fraction of the error removed per step */
    float gyro_drift_deg_per_min; /* game rotation vector heading drift (1-sigma) */
    float step_length_sigma;    /* fractional 1-sigma random error per step */
    float step_length_bias;     /* fractional 1-sigma systematic error (calibration,
                                   gait change when crouching in smoke) */
    float crawl_speed_mps;      /* assumed speed while moving without steps */
    float missed_motion_sigma_m_per_s; /* growth while moving without steps (crawling) */
    float nav_conf_warn;        /* below: speak "Way out is ..." */
    float nav_conf_unreliable;  /* below: "follow hose" warning */
    float nav_conf_scale_m;     /* position sigma at which confidence is ~0.5 */

    /* ---- Alerts ---- */
    uint32_t direction_repeat_ms;   /* repeat interval for "way out is ..." */
    uint32_t unreliable_repeat_ms;
    uint32_t person_alert_cooldown_ms;

    /* ---- Health supervisor ---- */
    uint32_t thermal_timeout_ms;
    uint32_t imu_timeout_ms;
    uint32_t detector_timeout_ms;

    /* ---- Power ---- */
    uint16_t batt_low_mv;
    uint16_t batt_critical_mv;
} ps_config_t;

void ps_config_default(ps_config_t *cfg);

#ifdef __cplusplus
}
#endif

#endif /* PS_CONFIG_H */
