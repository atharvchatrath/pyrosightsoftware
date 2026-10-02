/* Battery measurement maths and CPU-frequency selection (hardware independent, host-tested). */
#ifndef POWER_CALC_H
#define POWER_CALC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Battery voltage from the calibrated ADC pin voltage and a resistor divider
 * (battery -- r_top -- pin -- r_bottom -- GND). Saturates at 65535. */
uint16_t pwr_divider_to_batt_mv(int pin_mv, uint32_t r_top, uint32_t r_bottom);

/* Median of n (<= 15) samples; robust to the odd ADC spike from DSI/SPI noise. */
int pwr_median(const int *v, int n);

/* Largest supported ESP32-P4 CPU frequency <= want (min 40 = XTAL). */
uint16_t pwr_cpu_mhz_supported(uint16_t want, uint16_t chip_max);

#ifdef __cplusplus
}
#endif

#endif
