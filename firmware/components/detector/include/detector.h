/*
 * Neural detector: ESP-DL v3 dl::Model loaded from the "model" flash
 * partition (export from ml/ with esp-ppq, flash with tools/flash_partitions.sh).
 *
 * Model contract (ml/model.py):
 *   input  [1,120,160,1] int8   x = model_input code / 255, quantised with
 *                               the input tensor exponent (detector_quant.h)
 *   outputs, stride 4, 30 x 40 grid, 2 channels (fire, person):
 *     heat  class logits
 *     wh    box width/height in input pixels
 *     off   sub-cell centre offset 0..1
 *   Output tensors are matched by name ("heat"/"hm", "wh"/"size", "off"/"reg").
 *
 * The caller falls back to ps_system_run_classical() whenever detector_run()
 * fails or overruns its deadline.
 */
#ifndef DETECTOR_H
#define DETECTOR_H

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"
#include "pyrosight/ps_detect.h"
#include "pyrosight/ps_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DETECTOR_GRID_W 40
#define DETECTOR_GRID_H 30
#define DETECTOR_STRIDE 4

typedef struct {
    bool loaded;
    int input_exponent;
    uint32_t runs, failures, overruns;
    uint32_t last_us, max_us;
    float avg_us;                 /* exponential moving average */
} detector_stats_t;

/* Loads the model; ESP_ERR_NOT_FOUND if the partition is missing/erased,
 * ESP_ERR_INVALID_RESPONSE if the model does not match the contract. */
esp_err_t detector_init(const char *partition_label);
bool detector_ready(void);

/*
 * Run on a private copy of ps_system_t.model_input. On success fills `out`
 * (source = PS_DETECTOR_NEURAL, latency_us = inference time). Returns
 * ESP_ERR_TIMEOUT (out still filled) when inference took longer than
 * deadline_us, so the caller can prefer the classical result.
 */
esp_err_t detector_run(const ps_gray_frame_t *in, float score_threshold, uint32_t deadline_us,
                       ps_detections_t *out);

void detector_get_stats(detector_stats_t *out);

#ifdef __cplusplus
}
#endif

#endif /* DETECTOR_H */
