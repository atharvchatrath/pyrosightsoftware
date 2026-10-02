/*
 * Debounced short/long press classifier (hardware independent, host-tested).
 * Sample every few ms with the raw level; events come out as:
 *   BTN_EV_SHORT on release after a press shorter than long_ms;
 *   BTN_EV_LONG  as soon as the press has lasted long_ms (no wait for release,
 *                so the wearer gets immediate feedback); its release is silent.
 */
#ifndef BUTTON_FSM_H
#define BUTTON_FSM_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BTN_EV_NONE = 0,
    BTN_EV_SHORT,
    BTN_EV_LONG,
} btn_event_t;

typedef struct {
    uint16_t debounce_ms;   /* level must be stable this long (default 30) */
    uint16_t long_ms;       /* default 1500 */
    /* state */
    bool raw_last;
    uint32_t t_raw_change;
    bool stable;            /* debounced level (true = pressed) */
    uint32_t t_press;
    bool long_fired;
    bool primed;
} btn_fsm_t;

void btn_fsm_init(btn_fsm_t *b, uint16_t debounce_ms, uint16_t long_ms);
btn_event_t btn_fsm_update(btn_fsm_t *b, bool pressed, uint32_t t_ms);

#ifdef __cplusplus
}
#endif

#endif /* BUTTON_FSM_H */
