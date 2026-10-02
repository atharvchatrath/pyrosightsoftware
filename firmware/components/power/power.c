#include "power.h"

#include <stdbool.h>

#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_pm.h"
#include "power_calc.h"

static const char *TAG = "power";

#define SAMPLES 7

static struct {
    power_config_t cfg;
    adc_oneshot_unit_handle_t unit;
    adc_channel_t ch;
    adc_cali_handle_t cali;
    bool adc_ok;
    uint16_t cpu_mhz;
#if CONFIG_PM_ENABLE
    esp_pm_lock_handle_t busy;
#endif
} P;

static bool cali_init(adc_unit_t unit, adc_channel_t ch, adc_atten_t atten, adc_cali_handle_t *out)
{
#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    adc_cali_curve_fitting_config_t c = { .unit_id = unit, .chan = ch, .atten = atten, .bitwidth = ADC_BITWIDTH_DEFAULT };
    if (adc_cali_create_scheme_curve_fitting(&c, out) == ESP_OK) return true;
#endif
#if ADC_CALI_SCHEME_LINE_FITTING_SUPPORTED
    adc_cali_line_fitting_config_t l = { .unit_id = unit, .atten = atten, .bitwidth = ADC_BITWIDTH_DEFAULT };
    if (adc_cali_create_scheme_line_fitting(&l, out) == ESP_OK) return true;
#endif
    (void)unit; (void)ch; (void)atten; (void)out;
    return false;
}

esp_err_t power_init(const power_config_t *cfg)
{
    P.cfg = *cfg;
    P.cpu_mhz = cfg->chip_max_mhz;
    if (cfg->adc_gpio >= 0) {
        adc_unit_t unit;
        if (adc_oneshot_io_to_channel(cfg->adc_gpio, &unit, &P.ch) != ESP_OK) {
            ESP_LOGE(TAG, "GPIO%d is not ADC capable", cfg->adc_gpio);
        } else {
            adc_oneshot_unit_init_cfg_t u = { .unit_id = unit };
            if (adc_oneshot_new_unit(&u, &P.unit) == ESP_OK) {
                adc_oneshot_chan_cfg_t cc = { .atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_DEFAULT };
                P.adc_ok = adc_oneshot_config_channel(P.unit, P.ch, &cc) == ESP_OK;
                if (!cali_init(unit, P.ch, ADC_ATTEN_DB_12, &P.cali)) {
                    P.cali = NULL;
                    ESP_LOGW(TAG, "no ADC calibration scheme: battery reading is approximate");
                }
            }
        }
    }
#if CONFIG_PM_ENABLE
    esp_pm_lock_create(ESP_PM_CPU_FREQ_MAX, 0, "ps_busy", &P.busy);
#endif
    /* Start at the full clock; the policy may lower it later. */
    ps_power_policy_t full = { .cpu_mhz = cfg->chip_max_mhz };
    P.cpu_mhz = 0;
    power_apply_policy(&full);
    return ESP_OK;
}

uint16_t power_read_batt_mv(void)
{
    if (!P.adc_ok) return 0;
    int mv[SAMPLES], n = 0;
    for (int i = 0; i < SAMPLES; i++) {
        int raw;
        if (adc_oneshot_read(P.unit, P.ch, &raw) != ESP_OK) continue;
        int v;
        if (P.cali && adc_cali_raw_to_voltage(P.cali, raw, &v) == ESP_OK) mv[n++] = v;
        else mv[n++] = raw * 3300 / 4095;   /* uncalibrated estimate (12 dB, 12 bit) */
    }
    if (!n) return 0;
    return pwr_divider_to_batt_mv(pwr_median(mv, n), P.cfg.r_top_ohm, P.cfg.r_bottom_ohm);
}

esp_err_t power_apply_policy(const ps_power_policy_t *p)
{
    uint16_t mhz = pwr_cpu_mhz_supported(p->cpu_mhz ? p->cpu_mhz : P.cfg.chip_max_mhz, P.cfg.chip_max_mhz);
    if (mhz == P.cpu_mhz) return ESP_OK;
#if CONFIG_PM_ENABLE
    esp_pm_config_t pm = {
        .max_freq_mhz = mhz,
        .min_freq_mhz = 40,
        .light_sleep_enable = P.cfg.light_sleep,
    };
    esp_err_t e = esp_pm_configure(&pm);
    if (e != ESP_OK) {
        ESP_LOGW(TAG, "esp_pm_configure(%u MHz) failed: %s", mhz, esp_err_to_name(e));
        return e;
    }
    ESP_LOGI(TAG, "CPU ceiling %u MHz (DFS, light sleep %s)", mhz, P.cfg.light_sleep ? "on" : "off");
#else
    ESP_LOGI(TAG, "CONFIG_PM_ENABLE off: CPU stays at %u MHz (policy wants %u)", P.cfg.chip_max_mhz, mhz);
#endif
    P.cpu_mhz = mhz;
    return ESP_OK;
}

uint16_t power_cpu_mhz(void) { return P.cpu_mhz; }

void power_busy_begin(void)
{
#if CONFIG_PM_ENABLE
    if (P.busy) esp_pm_lock_acquire(P.busy);
#endif
}

void power_busy_end(void)
{
#if CONFIG_PM_ENABLE
    if (P.busy) esp_pm_lock_release(P.busy);
#endif
}
