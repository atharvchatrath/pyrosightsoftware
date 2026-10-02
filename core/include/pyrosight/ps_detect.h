/*
 * PyroSight core: detections, the classical fallback detector, NMS,
 * neural-network output decoding, and distance estimation.
 *
 * Two detector back-ends feed the same ps_detections_t:
 *   - the neural detector (ESP-DL on ESP32-P4; ml/ trains it), whose raw
 *     CenterNet-style output grids are decoded by ps_centernet_decode();
 *   - ps_detect_classical(): radiometric thresholds + connected components.
 *     It needs no model, so it is the boot default and the fallback if the
 *     model partition is empty or inference stalls.
 */
#ifndef PS_DETECT_H
#define PS_DETECT_H

#include "ps_config.h"
#include "ps_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PS_MAX_DETECTIONS 16

typedef enum {
    PS_CLASS_FIRE = 0,
    PS_CLASS_PERSON = 1,
    PS_NUM_CLASSES = 2,
} ps_class_t;

typedef enum {
    PS_DETECTOR_NONE = 0,
    PS_DETECTOR_CLASSICAL,
    PS_DETECTOR_NEURAL,
    PS_DETECTOR_REFERENCE,  /* simulation only: ground-truth boxes, for comparison */
} ps_detector_kind_t;

typedef struct {
    ps_class_t cls;
    float score;            /* 0..1 */
    float x, y, w, h;       /* box in thermal pixels, (x,y) = top-left */
    ps_dc_t peak_dc;        /* hottest pixel inside the box */
    /* Filled by ps_estimate_distances() for people. */
    float dist_m;           /* best estimate, 0 if unknown */
    float dist_min_m, dist_max_m;
    bool truncated;         /* touches the frame edge: distance is an upper bound */
} ps_detection_t;

typedef struct {
    ps_detection_t d[PS_MAX_DETECTIONS];
    uint8_t n;
    uint32_t frame_id;
    uint32_t t_ms;
    ps_detector_kind_t source;
    uint32_t latency_us;    /* capture -> detections ready */
} ps_detections_t;

/* Classical detector. `labels` is scratch of PS_THERM_PIXELS uint16. */
void ps_detect_classical(const ps_config_t *cfg, const ps_thermal_frame_t *frame,
                         uint16_t *labels, ps_detections_t *out);

/* Greedy per-class non-maximum suppression, in place, sorted by score. */
void ps_nms(ps_detections_t *dets, float iou_threshold);

float ps_iou(const ps_detection_t *a, const ps_detection_t *b);

/*
 * Decode CenterNet-style model output (see ml/model.py):
 *   heat[c][gy][gx]  class logits, c < PS_NUM_CLASSES
 *   wh[2][gy][gx]    box width/height in input pixels
 *   off[2][gy][gx]   sub-cell centre offset in [0,1)
 * Grid is grid_w x grid_h with `stride` input pixels per cell. Arrays are
 * dense float, channel-major (the int8 tensors are dequantised by the caller).
 * A cell is a detection if it is a 3x3 local maximum above the threshold.
 */
void ps_centernet_decode(const float *heat, const float *wh, const float *off,
                         int grid_w, int grid_h, int stride, float score_threshold,
                         ps_detections_t *out);

/* Rough person distance from apparent size (pinhole model). */
void ps_estimate_distances(const ps_config_t *cfg, ps_detections_t *dets);

/* Fill peak_dc for each detection from the radiometric frame. */
void ps_detect_fill_peaks(const ps_thermal_frame_t *frame, ps_detections_t *dets);

#ifdef __cplusplus
}
#endif

#endif /* PS_DETECT_H */
