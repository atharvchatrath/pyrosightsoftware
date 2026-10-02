/*
 * Glove-friendly two-button input, polled (no ISR: debouncing is time based
 * and a 10 ms poll costs nothing).
 *   A short -> PS_BTN_WHERE_OUT     A long (>1.5 s) -> PS_BTN_MARK_ENTRY
 *   B short -> PS_BTN_PALETTE       B long          -> PS_BTN_BRIGHTNESS
 */
#ifndef BUTTONS_H
#define BUTTONS_H

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"
#include "pyrosight/ps_system.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int pin_a, pin_b;
    bool active_low;        /* button to GND with internal pull-up */
    uint16_t debounce_ms;
    uint16_t long_ms;
} buttons_config_t;

esp_err_t buttons_init(const buttons_config_t *cfg);

/* Poll both buttons; returns the number of events written to out (max 2). */
int buttons_poll(uint32_t t_ms, ps_button_t out[2]);

#ifdef __cplusplus
}
#endif

#endif /* BUTTONS_H */
