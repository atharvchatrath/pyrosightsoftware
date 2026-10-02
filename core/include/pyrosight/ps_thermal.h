/*
 * PyroSight core: thermal image processing pipeline.
 *
 *   raw radiometric frame (deci-C)
 *     -> temporal denoise (motion-adaptive IIR)
 *     -> spatial denoise (3x3 median)
 *     -> [radiometric frame used by detection]
 *     -> display normalisation (percentile window + detail enhancement) -> gray8
 *     -> model input mapping (fixed absolute-temperature curve)          -> gray8
 */
#ifndef PS_THERMAL_H
#define PS_THERMAL_H

#include "ps_config.h"
#include "ps_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t acc[PS_THERM_PIXELS]; /* IIR state, Q4 fixed point deci-C */
    bool primed;
    /* Working buffers for display mapping (kept here to avoid stack use). */
    uint8_t base[PS_THERM_PIXELS];
    uint16_t hsum[PS_THERM_PIXELS];
    bool win_primed;
    /* Statistics from the last processed frame. */
    ps_dc_t min_dc, max_dc;
    ps_dc_t win_lo_dc, win_hi_dc; /* display window actually used */
} ps_thermal_t;

void ps_thermal_init(ps_thermal_t *t);

/* Denoise in place. `scratch` must hold PS_THERM_PIXELS values. */
void ps_thermal_denoise(ps_thermal_t *t, const ps_config_t *cfg,
                        ps_thermal_frame_t *frame, ps_dc_t *scratch);

/*
 * Map a denoised frame to 8-bit for display. The window follows the scene
 * (percentiles) but is smoothed over time so the image does not pump when a
 * fire enters the view. A detail-enhancement pass (unsharp mask against a
 * 5x5 box blur) keeps edges, door frames and people readable in flat scenes.
 */
void ps_thermal_to_display(ps_thermal_t *t, const ps_config_t *cfg,
                           const ps_thermal_frame_t *frame, ps_gray_frame_t *out);

/*
 * Map a frame to the fixed curve the neural detector is trained on. Absolute,
 * not scene-adaptive, so the model sees the same value for 37 C everywhere:
 *   -20 C .. 60 C  -> 0 .. 191   (~0.42 C per code: people, walls, smoke)
 *    60 C .. 600 C -> 192 .. 255 (fire, hot gas layer)
 * ps_thermal_model_code() is the per-pixel version, shared with ml/ (Python).
 */
uint8_t ps_thermal_model_code(ps_dc_t dc);
void ps_thermal_to_model_input(const ps_thermal_frame_t *frame, ps_gray_frame_t *out);

/* Percentile of a frame, computed with a 1 C bin histogram. */
ps_dc_t ps_thermal_percentile(const ps_thermal_frame_t *frame, uint8_t pct);

#ifdef __cplusplus
}
#endif

#endif /* PS_THERMAL_H */
