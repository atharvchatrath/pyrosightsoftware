/*
 * FLIR Lepton 3.5 ESP-IDF driver: transport glue around the hardware
 * independent protocol code (lepton_vospi_proto.c, lepton_cci.c).
 */
#include "lepton.h"

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

static const char *TAG = "lepton";

/* Packets per SPI transaction. 10 x 164 B = 1640 B, ~0.7 ms at 20 MHz. */
#define BATCH_PKTS 10
#define BATCH_BYTES (BATCH_PKTS * LEP_PKT_BYTES)
/* Upper bound of packets read per VSYNC before giving up on this pulse
 * (a segment is 60 packets; leading discard packets are allowed). */
#define MAX_PKTS_PER_VSYNC 120
/* No complete frame for this long -> resync; several in a row -> CCI reboot. */
#define LEPTON_STALL_MS 1000
#define LEPTON_STALLS_BEFORE_REBOOT 5
#define VSYNC_WAIT_MS 30   /* segments come every ~9.5 ms */

struct lepton_dev {
    lepton_config_t cfg;
    spi_device_handle_t spi;
    i2c_master_dev_handle_t i2c;
    lep_cci_t cci;
    SemaphoreHandle_t vsync_sem;
    uint8_t *rx;               /* DMA buffer */
    lep_asm_t *as;             /* ~48 KB, PSRAM is fine */
    lepton_stats_t st;
    uint32_t frame_id;
    int64_t t_last_frame_us;
    int consecutive_stalls;
    bool vsync_seen;
};

/* ---------- CCI transport (16-bit big-endian registers) ---------- */

static int cci_read(void *ctx, uint16_t reg, uint16_t *val)
{
    lepton_dev_t *d = ctx;
    uint8_t a[2] = { (uint8_t)(reg >> 8), (uint8_t)reg }, v[2];
    if (i2c_master_transmit_receive(d->i2c, a, 2, v, 2, 100) != ESP_OK) return -1;
    *val = (uint16_t)((v[0] << 8) | v[1]);
    return 0;
}

static int cci_write(void *ctx, uint16_t reg, uint16_t val)
{
    lepton_dev_t *d = ctx;
    uint8_t b[4] = { (uint8_t)(reg >> 8), (uint8_t)reg, (uint8_t)(val >> 8), (uint8_t)val };
    return i2c_master_transmit(d->i2c, b, 4, 100) == ESP_OK ? 0 : -1;
}

static void cci_delay(void *ctx, uint32_t ms)
{
    lepton_dev_t *d = ctx;
    if (d->cfg.keepalive) d->cfg.keepalive();
    TickType_t t = pdMS_TO_TICKS(ms);
    vTaskDelay(t ? t : 1);
}

/* ---------- VSYNC ---------- */

static void IRAM_ATTR vsync_isr(void *arg)
{
    lepton_dev_t *d = arg;
    BaseType_t woken = pdFALSE;
    d->st.vsync_irqs++;
    xSemaphoreGiveFromISR(d->vsync_sem, &woken);
    if (woken) portYIELD_FROM_ISR();
}

/* ---------- helpers ---------- */

static void hw_reset(lepton_dev_t *d)
{
    if (d->cfg.pin_pwr_dn >= 0) gpio_set_level(d->cfg.pin_pwr_dn, 1);
    if (d->cfg.pin_reset >= 0) {
        gpio_set_level(d->cfg.pin_reset, 0);
        vTaskDelay(pdMS_TO_TICKS(10));
        gpio_set_level(d->cfg.pin_reset, 1);
    }
}

static esp_err_t configure_camera(lepton_dev_t *d)
{
    /* The Lepton boots in ~1-2 s after reset; FFC at boot adds ~0.5 s. */
    if (lep_cci_wait_boot(&d->cci, 5000) != LEP_CCI_OK) {
        ESP_LOGE(TAG, "CCI: camera did not report booted");
        return ESP_ERR_TIMEOUT;
    }
    int r = lep_cci_setup_pyrosight(&d->cci, &d->st.setup);
    d->st.setup_ok = (r == 0);
    const lep_setup_report_t *s = &d->st.setup;
    ESP_LOGI(TAG, "setup: agc_off=%d rad=%d low_gain=%d tlinear=%d res0.1K=%d telem_off=%d vsync=%d (last result %d)",
             s->agc_off, s->radiometry_on, s->low_gain, s->tlinear_on, s->res_0_1k, s->telemetry_off,
             s->vsync_on, lep_cci_stats.last_result);
    if (!s->tlinear_on || !s->radiometry_on)
        ESP_LOGE(TAG, "TLinear/radiometry not enabled: temperatures will be WRONG");
    if (!s->res_0_1k)
        ESP_LOGW(TAG, "TLinear resolution not 0.1 K: hot scenes saturate, assuming 0.1 K anyway");
    return ESP_OK;
}

/* Idle CS (no SCLK) > 185 ms so the Lepton restarts its VoSPI stream. */
static void resync(lepton_dev_t *d)
{
    if (d->cfg.keepalive) d->cfg.keepalive();
    vTaskDelay(pdMS_TO_TICKS(LEP_RESYNC_IDLE_MS));
    if (d->cfg.keepalive) d->cfg.keepalive();
    lep_asm_resynced(d->as);
    xSemaphoreTake(d->vsync_sem, 0);  /* drop a stale pulse */
}

/* ---------- public API ---------- */

esp_err_t lepton_init(const lepton_config_t *cfg, lepton_dev_t **out)
{
    esp_err_t ret = ESP_OK;
    lepton_dev_t *d = heap_caps_calloc(1, sizeof(*d), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    ESP_RETURN_ON_FALSE(d, ESP_ERR_NO_MEM, TAG, "no mem");
    d->cfg = *cfg;
    d->as = heap_caps_calloc(1, sizeof(lep_asm_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!d->as) d->as = heap_caps_calloc(1, sizeof(lep_asm_t), MALLOC_CAP_8BIT);
    d->rx = heap_caps_aligned_calloc(64, 1, (BATCH_BYTES + 63) & ~63, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    d->vsync_sem = xSemaphoreCreateBinary();
    ESP_GOTO_ON_FALSE(d->as && d->rx && d->vsync_sem, ESP_ERR_NO_MEM, fail, TAG, "no mem");
    lep_asm_init(d->as, cfg->check_crc);

    /* Control pins. */
    uint64_t outs = 0;
    if (cfg->pin_reset >= 0) outs |= 1ULL << cfg->pin_reset;
    if (cfg->pin_pwr_dn >= 0) outs |= 1ULL << cfg->pin_pwr_dn;
    if (outs) {
        gpio_config_t io = { .pin_bit_mask = outs, .mode = GPIO_MODE_OUTPUT };
        ESP_GOTO_ON_ERROR(gpio_config(&io), fail, TAG, "gpio");
        hw_reset(d);
    }

    /* CCI. */
    i2c_device_config_t dc = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = LEP_CCI_I2C_ADDR,
        .scl_speed_hz = cfg->i2c_hz ? cfg->i2c_hz : 400000,
    };
    ESP_GOTO_ON_ERROR(i2c_master_bus_add_device(cfg->i2c_bus, &dc, &d->i2c), fail, TAG, "i2c add");
    d->cci = (lep_cci_t){ .read_reg = cci_read, .write_reg = cci_write, .delay_ms = cci_delay,
                          .ctx = d, .busy_timeout_ms = 1500 };

    /* VoSPI: mode 3 (CPOL=1, CPHA=1), receive only. */
    spi_bus_config_t bus = {
        .sclk_io_num = cfg->pin_sclk,
        .miso_io_num = cfg->pin_miso,
        .mosi_io_num = -1,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = BATCH_BYTES,
    };
    ESP_GOTO_ON_ERROR(spi_bus_initialize(cfg->spi_host, &bus, SPI_DMA_CH_AUTO), fail, TAG, "spi bus");
    spi_device_interface_config_t dev = {
        .mode = 3,
        .clock_speed_hz = cfg->spi_hz > 20000000 ? 20000000 : cfg->spi_hz,
        .spics_io_num = cfg->pin_cs,
        .queue_size = 1,
        .cs_ena_pretrans = 1,
        .cs_ena_posttrans = 1,
    };
    ESP_GOTO_ON_ERROR(spi_bus_add_device(cfg->spi_host, &dev, &d->spi), fail, TAG, "spi dev");

    if (configure_camera(d) != ESP_OK)
        ESP_LOGW(TAG, "continuing without CCI configuration (stream may be AGC/non-radiometric)");

    if (cfg->pin_vsync >= 0) {
        gpio_config_t io = {
            .pin_bit_mask = 1ULL << cfg->pin_vsync,
            .mode = GPIO_MODE_INPUT,
            .pull_down_en = GPIO_PULLDOWN_ENABLE,
            .intr_type = GPIO_INTR_POSEDGE,
        };
        ESP_GOTO_ON_ERROR(gpio_config(&io), fail, TAG, "vsync gpio");
        ret = gpio_install_isr_service(0);
        if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) goto fail;
        ret = ESP_OK;  /* already installed is fine */
        ESP_GOTO_ON_ERROR(gpio_isr_handler_add(cfg->pin_vsync, vsync_isr, d), fail, TAG, "isr");
    }

    /* Start from a clean VoSPI state. */
    resync(d);
    d->t_last_frame_us = esp_timer_get_time();
    *out = d;
    return ESP_OK;

fail:
    ESP_LOGE(TAG, "init failed: %s", esp_err_to_name(ret));
    if (d->vsync_sem) vSemaphoreDelete(d->vsync_sem);
    free(d->rx);
    free(d->as);
    free(d);
    return ret != ESP_OK ? ret : ESP_FAIL;
}

static esp_err_t read_batch(lepton_dev_t *d)
{
    spi_transaction_t t = {
        .length = BATCH_BYTES * 8,
        .rxlength = BATCH_BYTES * 8,
        .tx_buffer = NULL,
        .rx_buffer = d->rx,
    };
    esp_err_t e = spi_device_transmit(d->spi, &t);
    if (e != ESP_OK) d->st.spi_errors++;
    return e;
}

/*
 * Read the segment announced by one VSYNC pulse (or one poll). Returns true
 * when a full frame is ready in d->as->raw.
 */
static bool read_segment(lepton_dev_t *d)
{
    bool frame = false, seg_done = false;
    for (int n = 0; n < MAX_PKTS_PER_VSYNC; n += BATCH_PKTS) {
        if (read_batch(d) != ESP_OK) return false;
        bool all_discard = true;
        for (int i = 0; i < BATCH_PKTS; i++) {
            lep_feed_t r = lep_asm_feed(d->as, d->rx + i * LEP_PKT_BYTES);
            if (r != LEP_FEED_DISCARD) all_discard = false;
            if (r == LEP_FEED_FRAME) frame = true;
            if (r == LEP_FEED_SEGMENT || r == LEP_FEED_FRAME) seg_done = true;
        }
        if (d->as->need_resync) {
            resync(d);
            return false;
        }
        /* Done with this pulse once a segment completed and we are not in the
         * middle of the next one; a batch of pure discards means "not ready". */
        if (seg_done && !d->as->in_segment) break;
        if (all_discard && !d->as->in_segment) break;
    }
    return frame;
}

esp_err_t lepton_capture(lepton_dev_t *d, ps_thermal_frame_t *out, uint32_t timeout_ms)
{
    const int64_t t0 = esp_timer_get_time();
    for (;;) {
        const int64_t now = esp_timer_get_time();
        if ((now - t0) / 1000 >= timeout_ms) return ESP_ERR_TIMEOUT;

        bool pulse = true;
        if (d->cfg.pin_vsync >= 0) {
            pulse = xSemaphoreTake(d->vsync_sem, pdMS_TO_TICKS(VSYNC_WAIT_MS)) == pdTRUE;
            if (pulse) d->vsync_seen = true;
            else d->st.vsync_timeouts++;
        }
        if (!pulse && d->vsync_seen) {
            /* VSYNC normally runs continuously; a gap means the camera is busy
             * (FFC) or lost. Read anyway, the assembler copes with discards. */
        }
        const int64_t ts = esp_timer_get_time();
        const bool got = read_segment(d);
        if (!pulse || d->cfg.pin_vsync < 0) vTaskDelay(1);  /* polling: let others run */

        if (got) {
            lep_asm_to_frame(d->as, LEP_RES_0_1K, out);
            /* Count abandoned frames too, so ps_system sees the gap in frame_id. */
            d->frame_id = d->as->c.frames + d->as->c.frames_lost;
            out->frame_id = d->frame_id;
            out->t_ms = (uint32_t)(esp_timer_get_time() / 1000);
            d->st.last_capture_us = (uint32_t)(esp_timer_get_time() - ts);
            d->t_last_frame_us = esp_timer_get_time();
            d->consecutive_stalls = 0;
            return ESP_OK;
        }

        if ((esp_timer_get_time() - d->t_last_frame_us) / 1000 > LEPTON_STALL_MS) {
            d->st.stall_resyncs++;
            d->t_last_frame_us = esp_timer_get_time();
            if (++d->consecutive_stalls >= LEPTON_STALLS_BEFORE_REBOOT) {
                ESP_LOGW(TAG, "stream stalled repeatedly: rebooting camera");
                d->st.reboots++;
                d->consecutive_stalls = 0;
                hw_reset(d);
                if (d->cfg.pin_reset < 0) (void)lep_cci_run(&d->cci, LEP_CID_OEM_REBOOT);
                for (int i = 0; i < 15; i++) {
                    if (d->cfg.keepalive) d->cfg.keepalive();
                    vTaskDelay(pdMS_TO_TICKS(100));
                }
                configure_camera(d);
            } else {
                ESP_LOGW(TAG, "no frame for %d ms: resync", LEPTON_STALL_MS);
            }
            resync(d);
        }
    }
}

esp_err_t lepton_run_ffc(lepton_dev_t *d)
{
    return lep_cci_run_ffc(&d->cci) == LEP_CCI_OK ? ESP_OK : ESP_FAIL;
}

esp_err_t lepton_get_status(lepton_dev_t *d, lep_cam_status_t *st)
{
    return lep_cci_get_status(&d->cci, st, NULL) == LEP_CCI_OK ? ESP_OK : ESP_FAIL;
}

esp_err_t lepton_get_fpa_temp_dc(lepton_dev_t *d, int16_t *dc)
{
    return lep_cci_get_fpa_temp_dc(&d->cci, dc) == LEP_CCI_OK ? ESP_OK : ESP_FAIL;
}

void lepton_get_stats(lepton_dev_t *d, lepton_stats_t *out)
{
    *out = d->st;
    out->vospi = d->as->c;
}
