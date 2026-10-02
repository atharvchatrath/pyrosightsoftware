/*
 * Battery monitor (ADC oneshot + curve/line-fitting calibration) and
 * application of the core's ps_power_policy_t to the chip: CPU frequency
 * ceiling through esp_pm (DFS). The other policy fields (inference fps,
 * display fps, IMU rate, brightness) are applied by the tasks that own them.
 *
 * Light sleep: the MIPI-DSI video stream needs its clocks continuously, so
 * automatic light sleep is only enabled when CONFIG_PYROSIGHT_LIGHT_SLEEP is
 * set (for bench builds without a panel). FreeRTOS tickless idle still saves
 * power while tasks block.
 */
#ifndef POWER_H
#define POWER_H

#include <stdint.h>

#include "esp_err.h"
#include "pyrosight/ps_power.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int adc_gpio;             /* ADC-capable GPIO with the divided battery voltage, -1: none */
    uint32_t r_top_ohm, r_bottom_ohm;
    uint16_t chip_max_mhz;    /* CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ */
    bool light_sleep;
} power_config_t;

esp_err_t power_init(const power_config_t *cfg);

/* Battery millivolts (median of several conversions); 0 if not available. */
uint16_t power_read_batt_mv(void);

/* Apply the CPU part of a policy; no-op if unchanged. */
esp_err_t power_apply_policy(const ps_power_policy_t *p);

uint16_t power_cpu_mhz(void);

/*
 * With DFS the CPU idles at 40 MHz; heavy work (frame pipeline, inference,
 * rendering) runs between power_busy_begin()/end() at the policy ceiling.
 * Nestable and thread-safe (esp_pm lock); no-ops without CONFIG_PM_ENABLE.
 */
void power_busy_begin(void);
void power_busy_end(void);

#ifdef __cplusplus
}
#endif

#endif
