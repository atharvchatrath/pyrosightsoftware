#include "pyrosight/ps_power.h"

#include <string.h>

void ps_power_init(ps_power_t *p)
{
    memset(p, 0, sizeof(*p));
    p->mode = PS_PWR_NORMAL;
    p->pct = 100;
}

uint8_t ps_battery_pct(uint16_t mv)
{
    /* Typical 1S Li-ion discharge curve under light load. */
    static const uint16_t v[] = { 3300, 3450, 3600, 3700, 3750, 3800, 3900, 4000, 4100, 4200 };
    static const uint8_t pc[] = {    0,    5,   10,   20,   30,   45,   60,   75,   90,  100 };
    if (mv <= v[0]) return 0;
    for (int i = 1; i < 10; i++)
        if (mv <= v[i]) return (uint8_t)(pc[i - 1] + (pc[i] - pc[i - 1]) * (mv - v[i - 1]) / (v[i] - v[i - 1]));
    return 100;
}

bool ps_power_update(ps_power_t *p, const ps_config_t *cfg, uint16_t mv)
{
    /* Low-pass: load spikes (OLED, speaker) must not trigger a mode change. */
    if (!p->primed) { p->mv_filt = mv; p->primed = true; }
    else p->mv_filt = (uint16_t)(p->mv_filt + ((int)mv - (int)p->mv_filt) / 8);
    p->pct = ps_battery_pct(p->mv_filt);

    ps_power_mode_t m = p->mode;
    const uint16_t hyst = 50; /* mV, recovery needs a clear margin */
    if (p->mv_filt <= cfg->batt_critical_mv) m = PS_PWR_CRITICAL;
    else if (p->mv_filt <= cfg->batt_low_mv) { if (m != PS_PWR_CRITICAL || p->mv_filt > cfg->batt_critical_mv + hyst) m = PS_PWR_ECO; }
    else if (p->mv_filt > cfg->batt_low_mv + hyst) m = PS_PWR_NORMAL;
    bool changed = m != p->mode;
    p->mode = m;
    return changed;
}

void ps_power_policy(const ps_power_t *p, ps_power_policy_t *o)
{
    switch (p->mode) {
    case PS_PWR_NORMAL:   *o = (ps_power_policy_t){ 16, 255, 100, 30, 400 }; break;
    case PS_PWR_ECO:      *o = (ps_power_policy_t){  8, 160,  50, 20, 360 }; break;
    case PS_PWR_CRITICAL: *o = (ps_power_policy_t){  4, 110,  50, 15, 360 }; break;
    }
}
