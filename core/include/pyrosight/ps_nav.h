/*
 * PyroSight core: pedestrian dead reckoning and the way back out.
 *
 * Inputs come from the Bosch BNO085:
 *   - Game Rotation Vector (gyro + accel fusion, no magnetometer, because
 *     steel structure and fire make magnetic heading useless indoors);
 *   - Step Detector events;
 *   - Linear acceleration magnitude (to notice motion without steps, e.g.
 *     crawling, which the step detector does not count).
 *
 * The wearer marks the doorway on entry (button). From then on each step
 * advances the position by the calibrated step length along the current
 * heading. Turns and every few metres drop breadcrumbs; the way out retraces
 * the breadcrumbs in reverse, because the straight-line bearing to the door
 * usually goes through walls. Both bearings are reported.
 *
 * Position uncertainty grows with distance walked, heading drift over time
 * and untracked motion. Confidence in [0,1] is derived from it and drives the
 * audio alerts ("Way out is ...", then "follow hose").
 */
#ifndef PS_NAV_H
#define PS_NAV_H

#include "ps_config.h"
#include "ps_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PS_NAV_MAX_CRUMBS 128
#define PS_NAV_YAW_HISTORY 32

typedef enum {
    PS_NAV_IDLE = 0,     /* entry not marked yet */
    PS_NAV_TRACKING,     /* dead reckoning from the doorway */
    PS_NAV_LOST,         /* IMU data lost: position frozen, confidence decays */
} ps_nav_state_t;

typedef struct {
    ps_vec2_t p;     /* metres, entry frame: +x = heading at entry, +y = left */
    uint32_t t_ms;
    float heading_sigma; /* heading uncertainty when the crumb was dropped */
} ps_crumb_t;

typedef struct {
    ps_nav_state_t state;

    /* Heading. yaw_raw is the IMU game-rotation-vector yaw; yaw_offset makes
     * the entry heading 0 and absorbs heading-snap corrections. */
    float yaw_raw;
    float yaw_offset;
    float yaw;              /* corrected heading, rad, (-pi, pi] */
    bool have_yaw;

    ps_vec2_t pos;
    float dist_walked_m;
    uint32_t steps;
    uint32_t t_entry_ms;
    uint32_t t_last_ms;
    uint32_t t_last_step_ms;
    uint32_t t_last_imu_ms;

    /* Turn detection over a sliding window of yaw samples. */
    float yaw_hist[PS_NAV_YAW_HISTORY];
    uint32_t yaw_hist_t[PS_NAV_YAW_HISTORY];
    uint8_t yaw_hist_head, yaw_hist_n;
    uint32_t turns;
    uint32_t t_last_turn_ms;
    float last_turn_deg;    /* signed, + = left */

    /* Breadcrumbs: [0] is the doorway. */
    ps_crumb_t crumbs[PS_NAV_MAX_CRUMBS];
    uint16_t n_crumbs;
    uint16_t return_target;  /* index of the crumb the arrow points to */

    /* Uncertainty model (1-sigma, metres / radians). */
    float var_steps_m2;     /* independent per-step stride errors */
    /*
     * Unmeasured displacement, split by whether retracing cancels it.
     *
     *   var_crawl_m2 — crawling at an assumed speed. This is a speed-scale
     *     error of one wearer in one set of gear: it applies the same way on
     *     the route back, so most of it cancels on the retrace, exactly like
     *     the stride-length bias. Scaled by the remaining fraction of route.
     *   var_gap_m2 — displacement during an IMU outage. Nothing cancels this:
     *     it is a jump of unknown direction that the trail does not share.
     *
     * Lumping the two together is what made a 45 s crawl look as dangerous as
     * teleporting: the crawl term alone reached 6.8 m of sigma while the true
     * position error was 2.9 m, and the device sent the wearer to the hose.
     */
    float var_crawl_m2;
    float var_gap_m2;
    float crawl_dist_m;     /* total distance advanced without step events */

    /* Speed over the ground from recent step cadence, used to coast through
     * an IMU outage instead of assuming the wearer stood still. */
    float speed_mps;
    uint32_t t_lost_ms;     /* when the current outage began */
    uint32_t gaps;          /* IMU outages survived */
    float gap_coast_m;      /* distance coasted through outages */
    float cross_sigma_m;    /* cross-track error over the remaining route (derived) */
    float accel_lp;         /* low-passed linear acceleration magnitude */
    /* Crawl stride detection: a hand-knee cycle is an impact on the
     * accelerometer even though the walking step detector ignores it. */
    float accel_fast;       /* short-window average, for peak detection */
    float accel_base;       /* slow baseline the peak stands above */
    bool  crawl_armed;      /* fell back below the threshold since the last peak */
    uint8_t crawl_lock;     /* consecutive impacts at a plausible crawl cadence */
    uint32_t t_last_crawl_stride_ms;
    uint32_t crawl_strides;
    float heading_sigma_rad;
    float untracked_s;      /* seconds of motion without steps */
    float pos_sigma_m;
    float confidence;
} ps_nav_t;

typedef struct {
    bool valid;
    float route_bearing_rel_deg;  /* to the next return breadcrumb, relative to
                                     where the wearer faces; + = to the left */
    float home_bearing_rel_deg;   /* straight line to the doorway */
    float route_dist_m;           /* remaining path length via breadcrumbs */
    float home_dist_m;            /* straight-line distance */
    float pos_sigma_m;
    float confidence;
    bool exit_is_next;            /* the doorway is the next point on the route */
} ps_nav_guidance_t;

void ps_nav_init(ps_nav_t *nav);

/* Mark the doorway: resets position, breadcrumbs and uncertainty. */
void ps_nav_mark_entry(ps_nav_t *nav, uint32_t t_ms);

/* Heading sample from the game rotation vector, radians. */
void ps_nav_on_yaw(ps_nav_t *nav, const ps_config_t *cfg, float yaw_rad, uint32_t t_ms);

/* One step detected. */
void ps_nav_on_step(ps_nav_t *nav, const ps_config_t *cfg, uint32_t t_ms);

/* Linear acceleration magnitude (m/s^2) for untracked-motion detection. */
void ps_nav_on_linear_accel(ps_nav_t *nav, const ps_config_t *cfg, float mag, uint32_t t_ms);

/* Advance time: heading drift, IMU loss detection, confidence. */
void ps_nav_tick(ps_nav_t *nav, const ps_config_t *cfg, uint32_t t_ms);

void ps_nav_guidance(const ps_nav_t *nav, ps_nav_guidance_t *g);

/* Helpers shared with the firmware's BNO085 driver and the tests. */
float ps_wrap_pi(float a);
float ps_quat_to_yaw(float qw, float qx, float qy, float qz);

#ifdef __cplusplus
}
#endif

#endif /* PS_NAV_H */
