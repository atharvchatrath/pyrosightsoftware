#include "button_fsm.h"

#include <string.h>

void btn_fsm_init(btn_fsm_t *b, uint16_t debounce_ms, uint16_t long_ms)
{
    memset(b, 0, sizeof(*b));
    b->debounce_ms = debounce_ms ? debounce_ms : 30;
    b->long_ms = long_ms ? long_ms : 1500;
}

btn_event_t btn_fsm_update(btn_fsm_t *b, bool pressed, uint32_t t)
{
    if (!b->primed) {
        /* A button held at boot must be released before it counts. */
        b->primed = true;
        b->raw_last = pressed;
        b->stable = pressed;
        b->t_raw_change = t;
        b->long_fired = pressed;
        b->t_press = t;
        return BTN_EV_NONE;
    }
    if (pressed != b->raw_last) {
        b->raw_last = pressed;
        b->t_raw_change = t;
    }
    btn_event_t ev = BTN_EV_NONE;
    if (b->stable != b->raw_last && (uint32_t)(t - b->t_raw_change) >= b->debounce_ms) {
        b->stable = b->raw_last;
        if (b->stable) {
            /* Press time is the first edge, not the end of the debounce window. */
            b->t_press = b->t_raw_change;
            b->long_fired = false;
        } else {
            if (!b->long_fired) ev = BTN_EV_SHORT;
            b->long_fired = false;
        }
    }
    if (b->stable && !b->long_fired && (uint32_t)(t - b->t_press) >= b->long_ms) {
        b->long_fired = true;
        ev = BTN_EV_LONG;
    }
    return ev;
}
