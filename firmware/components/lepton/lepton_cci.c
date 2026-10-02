#include "lepton_cci.h"

#include <stddef.h>

lep_cci_stats_t lep_cci_stats;

static uint32_t busy_timeout(const lep_cci_t *c) { return c->busy_timeout_ms ? c->busy_timeout_ms : 1000; }

int lep_cci_wait_ready(const lep_cci_t *c)
{
    uint16_t st = 0;
    uint32_t waited = 0;
    for (;;) {
        if (c->read_reg(c->ctx, LEP_REG_STATUS, &st)) return LEP_CCI_ERR_IO;
        if (!(st & LEP_STATUS_BUSY)) return LEP_CCI_OK;
        if (waited >= busy_timeout(c)) { lep_cci_stats.timeouts++; return LEP_CCI_ERR_TIMEOUT; }
        c->delay_ms(c->ctx, 2);
        waited += 2;
    }
}

int lep_cci_wait_boot(const lep_cci_t *c, uint32_t timeout_ms)
{
    uint16_t st = 0;
    for (uint32_t waited = 0;; waited += 20) {
        if (!c->read_reg(c->ctx, LEP_REG_STATUS, &st) &&
            (st & (LEP_STATUS_BOOT_MODE | LEP_STATUS_BOOT_STATUS)) == (LEP_STATUS_BOOT_MODE | LEP_STATUS_BOOT_STATUS) &&
            !(st & LEP_STATUS_BUSY))
            return LEP_CCI_OK;
        if (waited >= timeout_ms) return LEP_CCI_ERR_TIMEOUT;
        c->delay_ms(c->ctx, 20);
    }
}

static int finish(const lep_cci_t *c)
{
    int r = lep_cci_wait_ready(c);
    if (r) { lep_cci_stats.errors++; return r; }
    uint16_t st;
    if (c->read_reg(c->ctx, LEP_REG_STATUS, &st)) { lep_cci_stats.errors++; return LEP_CCI_ERR_IO; }
    lep_cci_stats.last_result = (int8_t)(st >> 8);
    if (lep_cci_stats.last_result != 0) { lep_cci_stats.errors++; return LEP_CCI_ERR_CAMERA; }
    return LEP_CCI_OK;
}

static int issue(const lep_cci_t *c, uint16_t cmd, uint16_t n_words)
{
    lep_cci_stats.commands++;
    if (c->write_reg(c->ctx, LEP_REG_DATALEN, n_words)) return LEP_CCI_ERR_IO;
    if (c->write_reg(c->ctx, LEP_REG_COMMAND, cmd)) return LEP_CCI_ERR_IO;
    return finish(c);
}

int lep_cci_get(const lep_cci_t *c, uint16_t base, uint16_t *w, uint16_t n)
{
    if (n > 16) return LEP_CCI_ERR_IO;
    int r = lep_cci_wait_ready(c);
    if (r) return r;
    r = issue(c, lep_cci_cmd(base, LEP_TYPE_GET), n);
    if (r) return r;
    for (uint16_t i = 0; i < n; i++)
        if (c->read_reg(c->ctx, (uint16_t)(LEP_REG_DATA0 + 2 * i), &w[i])) return LEP_CCI_ERR_IO;
    return LEP_CCI_OK;
}

int lep_cci_set(const lep_cci_t *c, uint16_t base, const uint16_t *w, uint16_t n)
{
    if (n > 16) return LEP_CCI_ERR_IO;
    int r = lep_cci_wait_ready(c);
    if (r) return r;
    for (uint16_t i = 0; i < n; i++)
        if (c->write_reg(c->ctx, (uint16_t)(LEP_REG_DATA0 + 2 * i), w[i])) return LEP_CCI_ERR_IO;
    return issue(c, lep_cci_cmd(base, LEP_TYPE_SET), n);
}

int lep_cci_run(const lep_cci_t *c, uint16_t base)
{
    int r = lep_cci_wait_ready(c);
    if (r) return r;
    return issue(c, lep_cci_cmd(base, LEP_TYPE_RUN), 0);
}

int lep_cci_get_u32(const lep_cci_t *c, uint16_t base, uint32_t *v)
{
    uint16_t w[2];
    int r = lep_cci_get(c, base, w, 2);
    if (!r) *v = (uint32_t)w[0] | ((uint32_t)w[1] << 16);  /* DATA0 = LSW */
    return r;
}

int lep_cci_set_u32(const lep_cci_t *c, uint16_t base, uint32_t v)
{
    uint16_t w[2] = { (uint16_t)(v & 0xFFFF), (uint16_t)(v >> 16) };
    return lep_cci_set(c, base, w, 2);
}

int lep_cci_set_u32_verified(const lep_cci_t *c, uint16_t base, uint32_t v)
{
    int r = lep_cci_set_u32(c, base, v);
    if (r) return r;
    uint32_t back = ~v;
    r = lep_cci_get_u32(c, base, &back);
    if (r) return r;
    return back == v ? LEP_CCI_OK : LEP_CCI_ERR_VERIFY;
}

int lep_cci_setup_pyrosight(const lep_cci_t *c, lep_setup_report_t *rep)
{
    lep_setup_report_t r = { 0 };
    /* Order matters: gain mode before TLinear resolution (the camera may
     * re-select resolution on a gain change), radiometry before TLinear. */
    r.agc_off = lep_cci_set_u32_verified(c, LEP_CID_AGC_ENABLE, 0) == 0;
    r.radiometry_on = lep_cci_set_u32_verified(c, LEP_CID_RAD_ENABLE, 1) == 0;
    r.low_gain = lep_cci_set_u32_verified(c, LEP_CID_SYS_GAIN_MODE, LEP_GAIN_LOW) == 0;
    (void)lep_cci_set_u32(c, LEP_CID_RAD_TLINEAR_AUTORES, 0); /* best effort, [check] */
    r.tlinear_on = lep_cci_set_u32_verified(c, LEP_CID_RAD_TLINEAR_EN, 1) == 0;
    r.res_0_1k = lep_cci_set_u32_verified(c, LEP_CID_RAD_TLINEAR_RES, 0) == 0;
    r.telemetry_off = lep_cci_set_u32_verified(c, LEP_CID_SYS_TELEMETRY_EN, 0) == 0;
    r.vsync_on = lep_cci_set_u32_verified(c, LEP_CID_OEM_GPIO_MODE, LEP_GPIO_MODE_VSYNC) == 0;
    if (rep) *rep = r;
    return (r.agc_off && r.radiometry_on && r.low_gain && r.tlinear_on && r.res_0_1k &&
            r.telemetry_off && r.vsync_on) ? 0 : -1;
}

int lep_cci_run_ffc(const lep_cci_t *c)
{
    return lep_cci_run(c, LEP_CID_SYS_RUN_FFC);
}

int lep_cci_get_status(const lep_cci_t *c, lep_cam_status_t *st, uint16_t *cmd_count)
{
    uint16_t w[4];
    int r = lep_cci_get(c, LEP_CID_SYS_STATUS, w, 4);
    if (r) return r;
    if (st) *st = (lep_cam_status_t)((uint32_t)w[0] | ((uint32_t)w[1] << 16));
    if (cmd_count) *cmd_count = w[2];
    return 0;
}

int lep_cci_get_fpa_temp_dc(const lep_cci_t *c, int16_t *dc)
{
    uint16_t k100;
    int r = lep_cci_get(c, LEP_CID_SYS_FPA_TEMP_K, &k100, 1);
    if (!r) *dc = (int16_t)(((int32_t)k100 - 27315) / 10);
    return r;
}
