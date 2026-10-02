/*
 * Synthetic thermal scene renderer for the PyroSight simulator.
 *
 * A 2.5D ray caster over a floor plan of axis-aligned walls, with a hot gas
 * layer under the ceiling, billboard objects (people, fire, hot pipes),
 * smoke attenuation and Lepton-like sensor noise. Output is a radiometric
 * frame in deci-Celsius plus ground-truth boxes, so the pipeline, detector
 * and HUD can be exercised without hardware.
 */
#ifndef PS_SIM_SCENE_H
#define PS_SIM_SCENE_H

#include "pyrosight/ps_detect.h"
#include "pyrosight/ps_types.h"

typedef struct { float x0, y0, x1, y1; } sim_wall_t;

typedef enum { OBJ_FIRE, OBJ_PERSON_STANDING, OBJ_PERSON_LYING, OBJ_HOT_PIPE } sim_obj_kind_t;

typedef struct {
    sim_obj_kind_t kind;
    float x, y;       /* floor position, metres */
    float yaw;        /* body orientation (lying people), rad */
    float width;      /* metres across */
    float height;     /* metres tall (from z0) */
    float z0;         /* base height */
} sim_obj_t;

typedef struct {
    const sim_wall_t *walls;
    int n_walls;
    const sim_obj_t *objs;
    int n_objs;
    float wall_height;     /* m */
    float ambient_c;       /* floor-level air/surfaces */
    float layer_c;         /* hot gas layer under the ceiling */
    float outside_c;       /* through the open door */
    float smoke;           /* 0 clear .. 1 dense */
    float noise_c;         /* per-pixel temporal noise, 1 sigma */
} sim_scene_t;

typedef struct {
    ps_detection_t gt[PS_MAX_DETECTIONS]; /* cls, x, y, w, h, dist_m */
    int n;
} sim_truth_t;

void sim_scene_seed(unsigned seed);
float sim_randn(void);
float sim_rand01(void);

/* Render what a head-mounted camera at (x, y, cam_h) facing `yaw` sees. */
void sim_render(const sim_scene_t *sc, float x, float y, float yaw, float cam_h,
                float t_s, ps_thermal_frame_t *out, sim_truth_t *truth);

/* True if moving from a to b crosses a wall; *bx,*by are adjusted to slide along it. */
bool sim_collide(const sim_scene_t *sc, float ax, float ay, float *bx, float *by);

#endif
