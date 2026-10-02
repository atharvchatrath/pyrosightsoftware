#include "power_calc.h"

uint16_t pwr_divider_to_batt_mv(int pin_mv, uint32_t r_top, uint32_t r_bottom)
{
    if (pin_mv <= 0 || r_bottom == 0) return 0;
    uint64_t mv = ((uint64_t)pin_mv * (r_top + r_bottom) + r_bottom / 2) / r_bottom;
    return mv > 65535 ? 65535 : (uint16_t)mv;
}

int pwr_median(const int *v, int n)
{
    int s[15];
    if (n <= 0) return 0;
    if (n > 15) n = 15;
    for (int i = 0; i < n; i++) {
        int x = v[i], j = i;
        while (j > 0 && s[j - 1] > x) { s[j] = s[j - 1]; j--; }
        s[j] = x;
    }
    return s[n / 2];
}

uint16_t pwr_cpu_mhz_supported(uint16_t want, uint16_t chip_max)
{
    /* CPU clock is derived from the 360 MHz (400 MHz on rev >= 3) CPLL with
     * integer dividers; these are the steps DFS uses. [check against the
     * esp_pm docs for your silicon revision] */
    static const uint16_t steps[] = { 400, 360, 200, 180, 120, 90, 40 };
    for (unsigned i = 0; i < sizeof(steps) / sizeof(steps[0]); i++)
        if (steps[i] <= want && steps[i] <= chip_max && (chip_max % steps[i] == 0 || steps[i] == 40))
            return steps[i];
    return 40;
}
