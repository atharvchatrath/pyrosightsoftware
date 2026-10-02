/*
 * The simulated world, stepped 10 ms at a time: floor plan, scenarios, the
 * simulated firefighter (search inbound, then walk out following only the
 * eyepiece arrow, or the hose once the device says it is unreliable), the
 * BNO085 model and the Lepton frame clock. Shared by the command-line
 * simulator (ps_sim.c) and the browser build (web/).
 *
 * Each sim_world_step() advances 10 ms. When it returns SIM_EV_FRAME a new
 * thermal frame has been fed to ps_system_on_frame(); the caller then runs a
 * detector of its choice (threshold, neural, or ground truth) and renders.
 */
#ifndef PS_SIM_WORLD_H
#define PS_SIM_WORLD_H

#include "pyrosight/ps_system.h"
#include "scene.h"

#define SIM_IMU_DT_MS 10
#define SIM_FRAME_DT_MS 115 /* Lepton 3.5: ~8.7 Hz */

typedef struct { float x, y; } sim_wp_t;

typedef struct {
    const char *name;
    sim_wp_t inbound[16];
    int n_inbound;
    float smoke;
    float layer_c;
    float gyro_drift_dps;    /* true gyro heading drift, deg/s */
    float step_miss;         /* probability a real step is not detected */
    float step_bias;         /* true stride = calibrated * (1 + bias) */
    int crawl_from, crawl_to;/* inbound leg indices crawled (no steps detected) */
    float cam_drop_s[2];     /* thermal camera outage window */
    float imu_drop_s[2];     /* IMU outage window */
    int laps;                /* repeat the search loop to accumulate drift */
} sim_scenario_t;

extern const sim_scenario_t SIM_SCENARIOS[];
extern const int SIM_N_SCENARIOS;
extern const sim_wall_t SIM_WALLS[];
extern const int SIM_N_WALLS;
extern const sim_obj_t SIM_OBJS[];
extern const int SIM_N_OBJS;

enum { SIM_EV_NONE = 0, SIM_EV_FRAME = 1, SIM_EV_DONE = 2 };
enum { SIM_PHASE_INBOUND = 0, SIM_PHASE_SCAN = 1, SIM_PHASE_OUTBOUND = 2 };

typedef struct {
    const sim_scenario_t *sc;
    sim_scene_t scene;
    ps_system_t *sys;
    float step_bias;
    unsigned seed;

    /* True state. World frame: door at the origin, +x into the building. */
    float tx, ty, tyaw;
    float gyro_err, yaw_world_offset;
    int phase, leg, lap;
    float step_phase, scan_turned, slide_yaw;
    int slide_steps;
    uint32_t t, next_frame, frame_id, t_out_start;
    bool hose_used, exited, done, cam_down;

    ps_thermal_frame_t raw;  /* last rendered frame (before denoise) */
    sim_truth_t truth;       /* ground-truth boxes for it */
} sim_world_t;

/* Returns false for an unknown scenario. smoke < 0 and step_bias < -98 keep
 * the scenario's own values. `sys` is initialised here. */
bool sim_world_init(sim_world_t *w, ps_system_t *sys, const char *scenario, unsigned seed,
                    float smoke, float step_bias, bool heading_snap);

int sim_world_step(sim_world_t *w);

/* Device position estimate in world coordinates. */
void sim_world_estimate(const sim_world_t *w, float *x, float *y);

/* Ground-truth boxes as detections (for a "perfect detector" reference). */
void sim_world_truth_detections(const sim_world_t *w, ps_detections_t *out);

#endif
