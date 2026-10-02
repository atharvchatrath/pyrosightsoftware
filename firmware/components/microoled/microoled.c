#include "microoled.h"

#include <string.h>

#include "driver/gpio.h"
#include "driver/ppa.h"
#include "esp_cache.h"
#include "esp_check.h"
#include "esp_ldo_regulator.h"
#include "esp_lcd_mipi_dsi.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "panel_init_seq.h"

static const char *TAG = "microoled";

static struct {
    microoled_config_t cfg;
    esp_ldo_channel_handle_t ldo;
    esp_lcd_dsi_bus_handle_t bus;
    esp_lcd_panel_io_handle_t io;
    esp_lcd_panel_handle_t panel;
    void *fb[2];
    size_t fb_bytes;
    int bpp;
    int back;                   /* index of the buffer we may draw into */
    ppa_client_handle_t ppa;
    SemaphoreHandle_t refresh_sem;
    microoled_stats_t st;
    int out_w, out_h, off_x, off_y;
} D;

static bool IRAM_ATTR on_refresh_done(esp_lcd_panel_handle_t panel, esp_lcd_dpi_panel_event_data_t *e, void *ctx)
{
    BaseType_t woken = pdFALSE;
    xSemaphoreGiveFromISR(D.refresh_sem, &woken);
    return woken == pdTRUE;
}

static esp_err_t send_table(const panel_init_cmd_t *t, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        ESP_RETURN_ON_ERROR(esp_lcd_panel_io_tx_param(D.io, t[i].cmd, t[i].len ? t[i].data : NULL, t[i].len),
                            TAG, "init cmd 0x%02x", t[i].cmd);
        if (t[i].delay_ms) vTaskDelay(pdMS_TO_TICKS(t[i].delay_ms));
    }
    return ESP_OK;
}

esp_err_t microoled_init(const microoled_config_t *cfg)
{
    D.cfg = *cfg;
    D.bpp = cfg->rgb888 ? 3 : 2;
    D.out_w = PS_DISP_W * cfg->scale;
    D.out_h = PS_DISP_H * cfg->scale;
    ESP_RETURN_ON_FALSE(D.out_w <= cfg->h_res && D.out_h <= cfg->v_res, ESP_ERR_INVALID_ARG, TAG,
                        "%dx scaled frame (%dx%d) larger than the panel", cfg->scale, D.out_w, D.out_h);
    D.off_x = (cfg->h_res - D.out_w) / 2;
    D.off_y = (cfg->v_res - D.out_h) / 2;
    D.refresh_sem = xSemaphoreCreateBinary();

    /* DSI PHY supply from the on-chip LDO. */
    if (cfg->phy_ldo_chan >= 0) {
        esp_ldo_channel_config_t lc = { .chan_id = cfg->phy_ldo_chan, .voltage_mv = cfg->phy_ldo_mv };
        ESP_RETURN_ON_ERROR(esp_ldo_acquire_channel(&lc, &D.ldo), TAG, "DSI PHY LDO");
    }
    if (cfg->pin_reset >= 0) {
        gpio_config_t o = { .pin_bit_mask = 1ULL << cfg->pin_reset, .mode = GPIO_MODE_OUTPUT };
        gpio_config(&o);
        gpio_set_level(cfg->pin_reset, 0);
        vTaskDelay(pdMS_TO_TICKS(10));
        gpio_set_level(cfg->pin_reset, 1);
        vTaskDelay(pdMS_TO_TICKS(20));
    }

    esp_lcd_dsi_bus_config_t bc = {
        .bus_id = 0,
        .num_data_lanes = (uint8_t)cfg->lanes,
        .phy_clk_src = MIPI_DSI_PHY_CLK_SRC_DEFAULT,
        .lane_bit_rate_mbps = (uint32_t)cfg->lane_mbps,
    };
    ESP_RETURN_ON_ERROR(esp_lcd_new_dsi_bus(&bc, &D.bus), TAG, "dsi bus");
    esp_lcd_dbi_io_config_t ioc = { .virtual_channel = 0, .lcd_cmd_bits = 8, .lcd_param_bits = 8 };
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_io_dbi(D.bus, &ioc, &D.io), TAG, "dbi io");

    /* Vendor init in command mode, before video. */
    ESP_RETURN_ON_ERROR(send_table(panel_init_pre_video, sizeof(panel_init_pre_video) / sizeof(panel_init_pre_video[0])),
                        TAG, "pre-video init");
    uint8_t pf = cfg->rgb888 ? 0x77 : 0x55;
    esp_lcd_panel_io_tx_param(D.io, DCS_SET_PIXEL_FORMAT, &pf, 1);

    esp_lcd_dpi_panel_config_t pc = {
        .virtual_channel = 0,
        .dpi_clk_src = MIPI_DSI_DPI_CLK_SRC_DEFAULT,
        .dpi_clock_freq_mhz = (uint32_t)cfg->dpi_clk_mhz,
        .pixel_format = cfg->rgb888 ? LCD_COLOR_PIXEL_FORMAT_RGB888 : LCD_COLOR_PIXEL_FORMAT_RGB565,
        .num_fbs = 2,
        .video_timing = {
            .h_size = (uint32_t)cfg->h_res,
            .v_size = (uint32_t)cfg->v_res,
            .hsync_pulse_width = (uint32_t)cfg->hsync_pw,
            .hsync_back_porch = (uint32_t)cfg->hbp,
            .hsync_front_porch = (uint32_t)cfg->hfp,
            .vsync_pulse_width = (uint32_t)cfg->vsync_pw,
            .vsync_back_porch = (uint32_t)cfg->vbp,
            .vsync_front_porch = (uint32_t)cfg->vfp,
        },
    };
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_dpi(D.bus, &pc, &D.panel), TAG, "dpi panel");
    ESP_RETURN_ON_ERROR(esp_lcd_dpi_panel_get_frame_buffer(D.panel, 2, &D.fb[0], &D.fb[1]), TAG, "fbs");
    D.fb_bytes = (size_t)cfg->h_res * cfg->v_res * D.bpp;
    for (int i = 0; i < 2; i++) {
        memset(D.fb[i], 0, D.fb_bytes);   /* black border around the scaled image */
        esp_cache_msync(D.fb[i], D.fb_bytes, ESP_CACHE_MSYNC_FLAG_DIR_C2M);
    }
    esp_lcd_dpi_panel_event_callbacks_t cbs = { .on_refresh_done = on_refresh_done };
    ESP_RETURN_ON_ERROR(esp_lcd_dpi_panel_register_event_callbacks(D.panel, &cbs, NULL), TAG, "callbacks");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_init(D.panel), TAG, "dpi start");   /* starts the video stream */
    ESP_RETURN_ON_ERROR(send_table(panel_init_post_video, sizeof(panel_init_post_video) / sizeof(panel_init_post_video[0])),
                        TAG, "post-video init");

    ppa_client_config_t ppc = { .oper_type = PPA_OPERATION_SRM, .max_pending_trans_num = 1 };
    ESP_RETURN_ON_ERROR(ppa_register_client(&ppc, &D.ppa), TAG, "ppa");

    /* Buffer 0 is being scanned out after init; draw into 1 first. */
    D.back = 1;
    xSemaphoreGive(D.refresh_sem);
    ESP_LOGI(TAG, "%dx%d panel, %d lanes @ %d Mbps, image %dx%d at (%d,%d), %s", cfg->h_res, cfg->v_res,
             cfg->lanes, cfg->lane_mbps, D.out_w, D.out_h, D.off_x, D.off_y, cfg->rgb888 ? "RGB888" : "RGB565");
    return ESP_OK;
}

esp_err_t microoled_present(const ps_fb_t *fb)
{
    const int64_t t0 = esp_timer_get_time();
    /* The back buffer is free once the previous flip has been latched. */
    if (xSemaphoreTake(D.refresh_sem, pdMS_TO_TICKS(60)) != pdTRUE) D.st.flip_timeouts++;

    ppa_srm_oper_config_t op = {
        .in = {
            .buffer = fb->px,
            .pic_w = PS_DISP_W,
            .pic_h = PS_DISP_H,
            .block_w = PS_DISP_W,
            .block_h = PS_DISP_H,
            .block_offset_x = 0,
            .block_offset_y = 0,
            .srm_cm = PPA_SRM_COLOR_MODE_RGB565,
        },
        .out = {
            .buffer = D.fb[D.back],
            .buffer_size = D.fb_bytes,
            .pic_w = (uint32_t)D.cfg.h_res,
            .pic_h = (uint32_t)D.cfg.v_res,
            .block_offset_x = (uint32_t)D.off_x,
            .block_offset_y = (uint32_t)D.off_y,
            .srm_cm = D.cfg.rgb888 ? PPA_SRM_COLOR_MODE_RGB888 : PPA_SRM_COLOR_MODE_RGB565,
        },
        .rotation_angle = PPA_SRM_ROTATION_ANGLE_0,
        .scale_x = (float)D.cfg.scale,
        .scale_y = (float)D.cfg.scale,
        .mirror_x = false,
        .mirror_y = false,
        .rgb_swap = false,
        .byte_swap = false,
        .mode = PPA_TRANS_MODE_BLOCKING,
    };
    esp_err_t e = ppa_do_scale_rotate_mirror(D.ppa, &op);
    D.st.last_ppa_us = (uint32_t)(esp_timer_get_time() - t0);
    if (e != ESP_OK) {
        D.st.ppa_errors++;
        xSemaphoreGive(D.refresh_sem);
        return e;
    }
    /* Passing one of the panel's own frame buffers makes the driver switch to
     * it at the next frame instead of copying. */
    e = esp_lcd_panel_draw_bitmap(D.panel, 0, 0, D.cfg.h_res, D.cfg.v_res, D.fb[D.back]);
    xSemaphoreTake(D.refresh_sem, 0);   /* only a refresh after this flip frees the other buffer */
    D.back ^= 1;
    D.st.frames++;
    D.st.last_present_us = (uint32_t)(esp_timer_get_time() - t0);
    return e;
}

esp_err_t microoled_set_brightness(uint8_t level)
{
    if (!D.io) return ESP_ERR_INVALID_STATE;
    /* DCS commands in video mode go out in the blanking interval (LP). */
    if (D.cfg.brightness_2bytes) {
        uint16_t v = (uint16_t)level * 0x3FF / 255;   /* 10-bit panels; adjust per datasheet */
        uint8_t p[2] = { (uint8_t)(v >> 8), (uint8_t)v };
        return esp_lcd_panel_io_tx_param(D.io, DCS_SET_BRIGHTNESS, p, 2);
    }
    return esp_lcd_panel_io_tx_param(D.io, DCS_SET_BRIGHTNESS, &level, 1);
}

void microoled_get_stats(microoled_stats_t *out) { *out = D.st; }
