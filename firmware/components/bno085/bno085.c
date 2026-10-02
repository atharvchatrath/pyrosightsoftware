#include "bno085.h"

#include <stdlib.h>
#include <string.h>

#include "driver/gpio.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "pyrosight/ps_nav.h"

static const char *TAG = "bno085";

struct bno085_dev {
    bno085_config_t cfg;
    bno085_handlers_t h;
    i2c_master_dev_handle_t i2c;
    SemaphoreHandle_t int_sem;
    uint8_t seq[SHTP_NUM_CHANNELS];
    uint8_t buf[SHTP_MAX_PACKET];
    bno085_stats_t st;
    bool need_config;
    bool configured;
    bool awaiting_reset;       /* reset issued, reset-complete not yet seen */
    int64_t t_int_us;          /* time of the interrupt that announced the data */
    int64_t t_last_report_us;
    int64_t t_reset_us;
};

static void IRAM_ATTR int_isr(void *arg)
{
    bno085_t *d = arg;
    BaseType_t woken = pdFALSE;
    d->t_int_us = esp_timer_get_time();
    d->st.int_irqs++;
    xSemaphoreGiveFromISR(d->int_sem, &woken);
    if (woken) portYIELD_FROM_ISR();
}

static esp_err_t shtp_send(bno085_t *d, uint8_t chan, const uint8_t *payload, size_t n)
{
    uint8_t pkt[32];
    size_t len = shtp_build_packet(pkt, sizeof(pkt), chan, d->seq[chan]++, payload, n);
    if (!len) return ESP_ERR_INVALID_SIZE;
    esp_err_t e = i2c_master_transmit(d->i2c, pkt, len, 50);
    if (e != ESP_OK) d->st.i2c_errors++;
    return e;
}

static esp_err_t set_feature(bno085_t *d, uint8_t id, uint16_t hz)
{
    uint8_t p[17];
    sh2_build_set_feature(p, id, hz ? 1000000u / hz : 0, 0, 0, 0);
    return shtp_send(d, SHTP_CHAN_CONTROL, p, sizeof(p));
}

static esp_err_t configure(bno085_t *d)
{
    esp_err_t e = ESP_OK;
    uint8_t pid[2];
    sh2_build_product_id_request(pid);
    shtp_send(d, SHTP_CHAN_CONTROL, pid, sizeof(pid));
    e |= set_feature(d, SH2_GAME_ROTATION_VECTOR, d->cfg.rv_rate_hz);
    e |= set_feature(d, SH2_LINEAR_ACCELERATION, d->cfg.accel_rate_hz);
    /* Step detector is event driven; the interval only bounds the latency. */
    e |= set_feature(d, SH2_STEP_DETECTOR, 50);
    d->st.configs++;
    d->need_config = false;
    d->configured = (e == ESP_OK);
    d->t_last_report_us = esp_timer_get_time();
    ESP_LOGI(TAG, "features enabled: GRV %u Hz, lin.acc %u Hz, step detector (%s)",
             d->cfg.rv_rate_hz, d->cfg.accel_rate_hz, e == ESP_OK ? "ok" : "I2C error");
    return e;
}

static void hub_reset(bno085_t *d)
{
    d->configured = false;
    d->need_config = true;   /* also set by the reset-complete message */
    memset(d->seq, 0, sizeof(d->seq));
    if (d->cfg.pin_rst >= 0) {
        gpio_set_level(d->cfg.pin_rst, 0);
        vTaskDelay(pdMS_TO_TICKS(10));
        gpio_set_level(d->cfg.pin_rst, 1);
    } else {
        uint8_t cmd = SH2_EXEC_CMD_RESET;
        shtp_send(d, SHTP_CHAN_EXECUTABLE, &cmd, 1);
    }
    d->awaiting_reset = true;
    d->t_reset_us = esp_timer_get_time();
    d->t_last_report_us = d->t_reset_us;
}

static void on_event(void *ctx, const sh2_event_t *ev)
{
    bno085_t *d = ctx;
    int64_t t_us = d->t_int_us + ev->t_offset_us;
    uint32_t t_ms = (uint32_t)(t_us / 1000);
    switch (ev->type) {
    case SH2_EV_GAME_RV:
        d->st.rv_reports++;
        d->t_last_report_us = esp_timer_get_time();
        if (d->h.on_yaw) d->h.on_yaw(d->h.ctx, ps_quat_to_yaw(ev->u.q.w, ev->u.q.x, ev->u.q.y, ev->u.q.z), t_ms);
        break;
    case SH2_EV_LINEAR_ACCEL:
        d->st.accel_reports++;
        d->t_last_report_us = esp_timer_get_time();
        if (d->h.on_linear_accel) d->h.on_linear_accel(d->h.ctx, sh2_vec_mag(ev), t_ms);
        break;
    case SH2_EV_STEP:
        d->st.steps++;
        if (d->h.on_step) d->h.on_step(d->h.ctx, (uint32_t)((t_us - ev->u.step.latency_us) / 1000));
        break;
    case SH2_EV_RESET_COMPLETE:
        ESP_LOGW(TAG, "hub reset complete: re-enabling features");
        memset(d->seq, 0, sizeof(d->seq));
        d->awaiting_reset = false;
        d->need_config = true;
        d->configured = false;
        break;
    case SH2_EV_PRODUCT_ID:
        d->st.sw_major = ev->u.pid.sw_major;
        d->st.sw_minor = ev->u.pid.sw_minor;
        d->st.sw_build = ev->u.pid.build;
        ESP_LOGI(TAG, "SH-2 firmware %u.%u.%u build %lu (reset cause %u)", ev->u.pid.sw_major,
                 ev->u.pid.sw_minor, ev->u.pid.patch, (unsigned long)ev->u.pid.build, ev->u.pid.reset_cause);
        break;
    case SH2_EV_FEATURE_RESPONSE:
        ESP_LOGD(TAG, "feature 0x%02x interval %lu us", ev->u.feature.report_id,
                 (unsigned long)ev->u.feature.interval_us);
        break;
    default:
        break;
    }
}

/* Read one SHTP packet; returns false when nothing was pending. */
static bool read_packet(bno085_t *d)
{
    uint8_t hb[SHTP_HDR_LEN];
    if (i2c_master_receive(d->i2c, hb, sizeof(hb), 50) != ESP_OK) { d->st.i2c_errors++; return false; }
    shtp_hdr_t h;
    if (!shtp_parse_header(hb, &h)) return false;
    /* The second read starts again with a header (a continuation header if
     * the hub split the cargo); take the cargo after it. Packets longer than
     * our buffer (only the advertisement) are cut: their tail arrives as a
     * continuation and is skipped below. */
    size_t want = h.len > SHTP_MAX_PACKET ? SHTP_MAX_PACKET : h.len;
    if (i2c_master_receive(d->i2c, d->buf, want, 50) != ESP_OK) { d->st.i2c_errors++; return false; }
    shtp_hdr_t h2;
    if (!shtp_parse_header(d->buf, &h2)) return false;
    size_t len = h2.len < want ? h2.len : want;
    if (h.cont && h2.cont && h2.chan == SHTP_CHAN_COMMAND) return true;   /* advertisement tail */
    sh2_parse_cargo(&d->st.parser, h2.chan, h2.seq, d->buf + SHTP_HDR_LEN, len - SHTP_HDR_LEN, on_event, d);
    return true;
}

esp_err_t bno085_init(const bno085_config_t *cfg, const bno085_handlers_t *h, bno085_t **out)
{
    esp_err_t ret = ESP_OK;
    ESP_RETURN_ON_FALSE(cfg->pin_int >= 0, ESP_ERR_INVALID_ARG, TAG, "INT pin required");
    bno085_t *d = heap_caps_calloc(1, sizeof(*d), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    ESP_RETURN_ON_FALSE(d, ESP_ERR_NO_MEM, TAG, "no mem");
    d->cfg = *cfg;
    if (!d->cfg.silence_ms) d->cfg.silence_ms = 1000;
    if (!d->cfg.rv_rate_hz) d->cfg.rv_rate_hz = 100;
    if (!d->cfg.accel_rate_hz) d->cfg.accel_rate_hz = 50;
    if (h) d->h = *h;
    sh2_parser_init(&d->st.parser);
    d->int_sem = xSemaphoreCreateBinary();
    ESP_GOTO_ON_FALSE(d->int_sem, ESP_ERR_NO_MEM, fail, TAG, "sem");

    i2c_device_config_t dc = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = cfg->addr ? cfg->addr : BNO085_I2C_ADDR,
        .scl_speed_hz = cfg->i2c_hz ? cfg->i2c_hz : 400000,
    };
    ESP_GOTO_ON_ERROR(i2c_master_bus_add_device(cfg->bus, &dc, &d->i2c), fail, TAG, "i2c add");

    if (cfg->pin_rst >= 0) {
        gpio_config_t o = { .pin_bit_mask = 1ULL << cfg->pin_rst, .mode = GPIO_MODE_OUTPUT };
        ESP_GOTO_ON_ERROR(gpio_config(&o), fail, TAG, "rst");
        gpio_set_level(cfg->pin_rst, 1);
    }
    gpio_config_t in = {
        .pin_bit_mask = 1ULL << cfg->pin_int,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .intr_type = GPIO_INTR_NEGEDGE,
    };
    ESP_GOTO_ON_ERROR(gpio_config(&in), fail, TAG, "int");
    ret = gpio_install_isr_service(0);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) goto fail;
    ret = ESP_OK;
    ESP_GOTO_ON_ERROR(gpio_isr_handler_add(cfg->pin_int, int_isr, d), fail, TAG, "isr");

    hub_reset(d);
    *out = d;
    return ESP_OK;

fail:
    if (d->int_sem) vSemaphoreDelete(d->int_sem);
    free(d);
    return ret != ESP_OK ? ret : ESP_FAIL;
}

esp_err_t bno085_service(bno085_t *d, uint32_t timeout_ms)
{
    bool got = xSemaphoreTake(d->int_sem, pdMS_TO_TICKS(timeout_ms)) == pdTRUE;
    /* Drain while INT is held low (several packets may be queued). Also read
     * once on timeout if INT is low: an edge may have been missed. */
    int guard = 0;
    while ((got || gpio_get_level(d->cfg.pin_int) == 0) && guard++ < 16) {
        if (!got) d->t_int_us = esp_timer_get_time();
        got = false;
        if (!read_packet(d)) break;
        if (gpio_get_level(d->cfg.pin_int) != 0) break;
    }
    /* Features are sent once the hub announced its reset (or 300 ms after
     * our reset if the message was missed). */
    const int64_t now = esp_timer_get_time();
    if (d->need_config && now - d->t_reset_us > 50000 &&
        (!d->awaiting_reset || now - d->t_reset_us > 300000)) {
        d->awaiting_reset = false;
        configure(d);
    }
    if (now - d->t_last_report_us > (int64_t)d->cfg.silence_ms * 1000) {
        ESP_LOGW(TAG, "no reports for %lu ms: resetting hub", (unsigned long)d->cfg.silence_ms);
        d->st.reinits++;
        hub_reset(d);
        return ESP_ERR_TIMEOUT;
    }
    return ESP_OK;
}

esp_err_t bno085_set_rv_rate(bno085_t *d, uint16_t hz)
{
    if (hz == d->cfg.rv_rate_hz) return ESP_OK;
    d->cfg.rv_rate_hz = hz;
    return d->configured ? set_feature(d, SH2_GAME_ROTATION_VECTOR, hz) : ESP_OK;
}

void bno085_get_stats(bno085_t *d, bno085_stats_t *out) { *out = d->st; }
