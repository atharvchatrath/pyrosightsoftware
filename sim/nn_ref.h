/*
 * Float reference runtime for the PyroSight detector (the .psnn format written
 * by ml/export_nnref.py). Used by the simulator and the browser demo so they
 * run the trained model without ESP-DL. Supports exactly the ops the model
 * uses: Conv (group 1), ReLU, nearest Resize, Add, Sigmoid.
 */
#ifndef PS_NN_REF_H
#define PS_NN_REF_H

#include <stddef.h>
#include <stdint.h>

#include "pyrosight/ps_types.h"

typedef struct nn_ref nn_ref_t;

/* The blob must outlive the model (weights are used in place). NULL on error. */
nn_ref_t *nn_ref_load(const uint8_t *blob, size_t len);
void nn_ref_free(nn_ref_t *m);

/* Run on model-input codes (ps_thermal_model_code, 160x120); the network sees
 * code/255. Outputs are dense [2][30][40] float arrays. Returns 0 on success. */
int nn_ref_run(nn_ref_t *m, const uint8_t *codes, float *heat, float *wh, float *off);

#endif
