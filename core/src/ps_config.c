#include "pyrosight/ps_config.h"

void ps_config_default(ps_config_t *c)
{
    c->temporal_shift = 2;            /* ~4-frame time constant at 8.7 fps */
    c->motion_threshold_dc = PS_DC(2.0);
    c->median3x3 = true;
    c->min_display_span_dc = PS_DC(8.0);
    c->lo_percentile = 1;
    c->hi_percentile = 99;
    c->detail_gain_q4 = 24;           /* 1.5x detail boost */

    c->fire_min_dc = PS_DC(150.0);
    c->person_min_dc = PS_DC(28.0);
    c->person_max_dc = PS_DC(40.0);
    c->person_contrast_dc = PS_DC(2.0);
    c->fire_min_area = 6;
    c->person_min_area = 12;
    c->nms_iou = 0.45f;
    c->score_threshold = 0.35f;

    c->hfov_deg = 57.0f;
    c->person_height_m = 1.7f;

    c->step_length_m = 0.62f;          /* shorter than normal gait: heavy gear */
    c->turn_threshold_deg = 50.0f;
    c->turn_window_ms = 2500.0f;
    c->breadcrumb_spacing_m = 4.0f;
    c->waypoint_reached_m = 1.5f;
    c->heading_snap = true;
    c->heading_snap_deg = 12.0f;
    c->heading_snap_gain = 0.08f;
    c->gyro_drift_deg_per_min = 1.5f;
    c->step_length_sigma = 0.10f;
    c->step_length_bias = 0.08f;
    c->crawl_speed_mps = 0.3f;
    c->missed_motion_sigma_m_per_s = 0.15f;
    c->nav_conf_warn = 0.6f;
    c->nav_conf_unreliable = 0.3f;
    c->nav_conf_scale_m = 4.0f;

    c->direction_repeat_ms = 15000;
    c->unreliable_repeat_ms = 20000;
    c->person_alert_cooldown_ms = 10000;

    c->thermal_timeout_ms = 600;     /* ~5 missed frames */
    c->imu_timeout_ms = 300;
    c->detector_timeout_ms = 1000;

    c->batt_low_mv = 3500;
    c->batt_critical_mv = 3350;
}
