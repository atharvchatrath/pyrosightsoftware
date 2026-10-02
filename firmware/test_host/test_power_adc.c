/* Host tests: battery divider maths, median filter, CPU frequency steps. */
#include "../../core/tests/ps_test.h"
#include "power_calc.h"

int main(void)
{
    CHECK(pwr_divider_to_batt_mv(2100, 100000, 100000) == 4200);
    CHECK(pwr_divider_to_batt_mv(1000, 200000, 100000) == 3000);
    CHECK(pwr_divider_to_batt_mv(0, 1, 1) == 0);
    CHECK(pwr_divider_to_batt_mv(100, 1, 0) == 0);
    CHECK(pwr_divider_to_batt_mv(3000, 1000000, 10) == 65535);
    int v[5] = { 2100, 2102, 900, 2101, 3300 };
    CHECK(pwr_median(v, 5) == 2101);
    int w[1] = { 7 };
    CHECK(pwr_median(w, 1) == 7);
    CHECK(pwr_median(v, 0) == 0);
    CHECK(pwr_cpu_mhz_supported(360, 360) == 360);
    CHECK(pwr_cpu_mhz_supported(240, 360) == 180);
    CHECK(pwr_cpu_mhz_supported(160, 360) == 120);
    CHECK(pwr_cpu_mhz_supported(100, 360) == 90);
    CHECK(pwr_cpu_mhz_supported(10, 360) == 40);
    CHECK(pwr_cpu_mhz_supported(400, 400) == 400);
    CHECK(pwr_cpu_mhz_supported(360, 400) == 200);
    TEST_MAIN_END();
}
