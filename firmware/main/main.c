/*
 * PyroSight firmware entry point: bring up the hardware, then start the tasks
 * (app_tasks.c). Every subsystem is optional at runtime: a missing sensor or
 * peripheral is logged and the device keeps running with what it has (the
 * core's health supervisor tells the wearer what is lost).
 */
#include <string.h>

#include "app.h"
#include "audio.h"
#include "board_pins.h"
#include "buttons.h"
#include "detector.h"
#include "driver/i2c_master.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "microoled.h"
#include "power.h"
#include "sdkconfig.h"

static const char *TAG = "pyrosight";

app_t g_app;

uint32_t app_now_ms(void) { return (uint32_t)(esp_timer_get_time() / 1000); }

void app_lock(void)
{
    int64_t t0 = esp_timer_get_time();
    xSemaphoreTake(g_app.lock, portMAX_DELAY);
    uint32_t w = (uint32_t)(esp_timer_get_time() - t0);
    if (w > g_app.lock_wait_max_us) g_app.lock_wait_max_us = w;
}

void app_unlock(void) { xSemaphoreGive(g_app.lock); }

/* The Lepton driver calls this during long resync/reboot waits; it runs in
 * the thermal task, which is subscribed to the task watchdog. */
static void lepton_keepalive(void)
{
    if (esp_task_wdt_status(NULL) == ESP_OK) esp_task_wdt_reset();   /* not subscribed during boot */
}

/* ---- IMU callbacks (run in the IMU task) ---- */

static void imu_yaw(void *ctx, float yaw, uint32_t t_ms)
{
    app_lock();
    ps_system_on_yaw(g_app.sys, yaw, t_ms);
    app_unlock();
}

static void imu_step(void *ctx, uint32_t t_ms)
{
    app_lock();
    ps_system_on_step(g_app.sys, t_ms);
    app_unlock();
}

static void imu_accel(void *ctx, float mag, uint32_t t_ms)
{
    app_lock();
    ps_system_on_linear_accel(g_app.sys, mag, t_ms);
    app_unlock();
}

static i2c_master_bus_handle_t new_bus(i2c_port_num_t port, int sda, int scl)
{
    i2c_master_bus_config_t c = {
        .i2c_port = port,
        .sda_io_num = sda,
        .scl_io_num = scl,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,   /* boards should still fit 2.2-4.7 k pull-ups */
    };
    i2c_master_bus_handle_t h = NULL;
    esp_err_t e = i2c_new_master_bus(&c, &h);
    if (e != ESP_OK) ESP_LOGE(TAG, "I2C%d init failed: %s", (int)port, esp_err_to_name(e));
    return h;
}

static void boot_screen(const char *line2)
{
    if (!g_app.display_ok) return;
    ps_fb_fill(g_app.fb, 0);
    const uint16_t amber = ps_rgb565(255, 160, 0);
    ps_fb_text(g_app.fb, (PS_DISP_W - ps_fb_text_width("PYROSIGHT", 3)) / 2, 90, "PYROSIGHT", 3, amber, true);
    ps_fb_text(g_app.fb, (PS_DISP_W - ps_fb_text_width(line2, 1)) / 2, 130, line2, 1, amber, true);
    microoled_present(g_app.fb);
}

void app_main(void)
{
    ESP_LOGI(TAG, "PyroSight starting (sizeof(ps_system_t) = %u)", (unsigned)sizeof(ps_system_t));

    g_app.sys = heap_caps_calloc(1, sizeof(ps_system_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    g_app.infer_in = heap_caps_calloc(1, sizeof(ps_gray_frame_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    g_app.fb = heap_caps_aligned_calloc(64, 1, sizeof(ps_fb_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_DMA);
    if (!g_app.fb) g_app.fb = heap_caps_aligned_calloc(64, 1, sizeof(ps_fb_t), MALLOC_CAP_SPIRAM);
    g_app.lock = xSemaphoreCreateMutex();
    if (!g_app.sys || !g_app.infer_in || !g_app.fb || !g_app.lock) {
        ESP_LOGE(TAG, "out of memory at boot (PSRAM enabled?)");
        abort();
    }
    ps_system_init(g_app.sys, NULL, app_now_ms());

    /* Power first: everything after runs at the policy's CPU ceiling. */
    power_config_t pc = {
        .adc_gpio = BOARD_BATT_ADC_GPIO,
        .r_top_ohm = CONFIG_PS_BATT_R_TOP,
        .r_bottom_ohm = CONFIG_PS_BATT_R_BOTTOM,
        .chip_max_mhz = CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
        .light_sleep = PS_CFG_LIGHT_SLEEP,
    };
    power_init(&pc);

    /* I2C buses. */
    i2c_master_bus_handle_t codec_bus = new_bus(BOARD_CODEC_I2C_PORT, BOARD_CODEC_I2C_SDA, BOARD_CODEC_I2C_SCL);
#if BOARD_SENSOR_I2C_SEPARATE
    i2c_master_bus_handle_t sensor_bus = new_bus(BOARD_SENSOR_I2C_PORT, BOARD_SENSOR_I2C_SDA, BOARD_SENSOR_I2C_SCL);
#else
    i2c_master_bus_handle_t sensor_bus = codec_bus;
#endif

    /* Display. */
    microoled_config_t dc = {
        .h_res = CONFIG_PS_OLED_H_RES, .v_res = CONFIG_PS_OLED_V_RES,
        .hsync_pw = CONFIG_PS_OLED_HSYNC_PW, .hbp = CONFIG_PS_OLED_HBP, .hfp = CONFIG_PS_OLED_HFP,
        .vsync_pw = CONFIG_PS_OLED_VSYNC_PW, .vbp = CONFIG_PS_OLED_VBP, .vfp = CONFIG_PS_OLED_VFP,
        .dpi_clk_mhz = CONFIG_PS_OLED_DPI_CLK_MHZ,
        .lanes = CONFIG_PS_OLED_LANES,
        .lane_mbps = CONFIG_PS_OLED_LANE_MBPS,
        .scale = CONFIG_PS_OLED_SCALE,
        .pin_reset = BOARD_OLED_RESET,
        .phy_ldo_chan = CONFIG_PS_DSI_PHY_LDO_CHAN,
        .phy_ldo_mv = CONFIG_PS_DSI_PHY_LDO_MV,
        .brightness_2bytes = PS_CFG_OLED_BRIGHTNESS_2BYTES,
        .rgb888 = PS_CFG_OLED_RGB888,
    };
    g_app.display_ok = microoled_init(&dc) == ESP_OK;
    if (g_app.display_ok) microoled_set_brightness(ps_system_brightness(g_app.sys));
    boot_screen("STARTING");

    /* Audio. */
    if (codec_bus) {
        audio_config_t ac = {
            .i2c_bus = codec_bus,
            .codec_addr_7bit = CONFIG_PS_CODEC_ADDR,
            .i2s_port = BOARD_I2S_PORT,
            .pin_mclk = BOARD_I2S_MCLK, .pin_bclk = BOARD_I2S_BCLK,
            .pin_ws = BOARD_I2S_WS, .pin_dout = BOARD_I2S_DOUT,
            .pin_pa = BOARD_PA_ENABLE,
            .volume = CONFIG_PS_AUDIO_VOLUME,
            .partition_label = "audio",
        };
        g_app.audio_ok = audio_init(&ac) == ESP_OK;
        if (g_app.audio_ok) audio_beep(PS_PRIO_INFO);
    }

    /* Thermal camera. */
    if (sensor_bus) {
        lepton_config_t lc = {
            .spi_host = BOARD_LEPTON_SPI_HOST,
            .pin_sclk = BOARD_LEPTON_SCLK, .pin_miso = BOARD_LEPTON_MISO, .pin_cs = BOARD_LEPTON_CS,
            .pin_vsync = BOARD_LEPTON_VSYNC,
            .pin_reset = BOARD_LEPTON_RESET, .pin_pwr_dn = BOARD_LEPTON_PWR_DN,
            .spi_hz = CONFIG_PS_LEPTON_SPI_HZ,
            .i2c_bus = sensor_bus,
            .i2c_hz = CONFIG_PS_SENSOR_I2C_HZ,
            .check_crc = PS_CFG_LEPTON_CHECK_CRC,
            .keepalive = lepton_keepalive,
        };
        if (lepton_init(&lc, &g_app.lepton) != ESP_OK) g_app.lepton = NULL;

        bno085_config_t ic = {
            .bus = sensor_bus,
            .addr = CONFIG_PS_BNO085_ADDR,
            .i2c_hz = CONFIG_PS_SENSOR_I2C_HZ,
            .pin_int = BOARD_BNO085_INT,
            .pin_rst = BOARD_BNO085_RST,
            .rv_rate_hz = g_app.sys->policy.imu_rate_hz,
            .accel_rate_hz = 50,
            .silence_ms = 1000,
        };
        bno085_handlers_t ih = { .on_yaw = imu_yaw, .on_step = imu_step, .on_linear_accel = imu_accel };
        if (bno085_init(&ic, &ih, &g_app.imu) != ESP_OK) g_app.imu = NULL;
    }
    if (!g_app.lepton) ESP_LOGE(TAG, "thermal camera unavailable");
    if (!g_app.imu) ESP_LOGE(TAG, "IMU unavailable: navigation disabled");

    /* Neural detector (classical fallback otherwise). */
#if CONFIG_PS_NEURAL_DETECTOR
    boot_screen("LOADING MODEL");
    g_app.neural_ok = detector_init("model") == ESP_OK;
#endif
    ESP_LOGI(TAG, "detector: %s", g_app.neural_ok ? "neural (ESP-DL)" : "classical");

    buttons_config_t bc = {
        .pin_a = BOARD_BUTTON_A, .pin_b = BOARD_BUTTON_B,
        .active_low = true, .debounce_ms = 30, .long_ms = CONFIG_PS_BUTTON_LONG_MS,
    };
    buttons_init(&bc);

    boot_screen(g_app.neural_ok ? "READY  AI" : "READY");
    app_start_tasks();
    ESP_LOGI(TAG, "running; free internal %u B, PSRAM %u B",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}
