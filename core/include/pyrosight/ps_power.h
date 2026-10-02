/*
 * PyroSight core: battery state and power policy.
 *
 * The policy is advisory: the firmware applies it (inference rate, panel
 * brightness cap, IMU report rate, CPU frequency). Detection and navigation
 * never switch off on low battery; the device degrades, it does not go dark.
 */
#ifndef PS_POWER_H
#define PS_POWER_H

#include "ps_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PS_PWR_NORMAL = 0,
    PS_PWR_ECO,       /* battery low: halve inference rate, cap brightness */
    PS_PWR_CRITICAL,  /* battery critical: minimum rates, keep warning the wearer */
} ps_power_mode_t;

typedef struct {
    uint8_t max_infer_fps;   /* detector runs at most this often */
    uint8_t max_brightness;  /* panel brightness cap 0..255 */
    uint16_t imu_rate_hz;    /* game rotation vector report rate */
    uint8_t display_fps;
    uint16_t cpu_mhz;        /* ESP32-P4 DFS ceiling */
} ps_power_policy_t;

typedef struct {
    ps_power_mode_t mode;
    uint16_t mv_filt;
    uint8_t pct;
    bool primed;
} ps_power_t;

void ps_power_init(ps_power_t *p);

/* Single-cell Li-ion open-circuit-ish voltage to percent (piecewise linear). */
uint8_t ps_battery_pct(uint16_t mv);

/* Feed a battery reading; returns true if the mode changed. */
bool ps_power_update(ps_power_t *p, const ps_config_t *cfg, uint16_t mv);

void ps_power_policy(const ps_power_t *p, ps_power_policy_t *out);

#ifdef __cplusplus
}
#endif

#endif /* PS_POWER_H */
