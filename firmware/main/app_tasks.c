/*
 * PyroSight tasks. See app.h for the task/core/priority layout and the
 * README for the timing budget.
 */
#include <string.h>

#include "app.h"
#include "audio.h"
#include "buttons.h"
#include "detector.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "microoled.h"
#include "power.h"
#include "sdkconfig.h"

static const char *TAG = "app";

/* Neural detector is benched for this long after repeated overruns. */
#define NEURAL_BENCH_MS 30000
#define NEURAL_MAX_CONSECUTIVE_OVERRUNS 3

static inline uint32_t us_since(int64_t t0) { return (uint32_t)(esp_timer_get_time() - t0); }

/* ------------------------------------------------------------------------ */
/* Thermal capture: VSYNC-driven, highest priority, core 0.                  */
/* ------------------------------------------------------------------------ */

static void thermal_task(void *arg)
{
    ps_thermal_frame_t *raw = heap_caps_malloc(sizeof(*raw), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!raw) raw = heap_caps_malloc(sizeof(*raw), MALLOC_CAP_8BIT);
    esp_task_wdt_add(NULL);
    for (;;) {
        esp_task_wdt_reset();
        /* Capture (SPI, ~5 ms per segment at 16 MHz) outside the lock. */
        if (lepton_capture(g_app.lepton, raw, 500) != ESP_OK) continue;  /* tick reports camera loss */

        power_busy_begin();
        const int64_t t0 = esp_timer_get_time();
        app_lock();
        ps_system_on_frame(g_app.sys, raw);
        /* Hand the model input to the inference task (private copy). */
        memcpy(g_app.infer_in, &g_app.sys->model_input, sizeof(ps_gray_frame_t));
        const uint32_t us = us_since(t0);
        g_app.sys->stats.last_pipeline_us = us;
        app_unlock();
        power_busy_end();
        if (us > g_app.pipeline_max_us) g_app.pipeline_max_us = us;

        xTaskNotify(g_app.infer_task, raw->frame_id, eSetValueWithOverwrite);
    }
}

/* ------------------------------------------------------------------------ */
/* Inference: core 1, once per new frame, capped by the power policy.        */
/* ------------------------------------------------------------------------ */

static void inference_task(void *arg)
{
    ps_gray_frame_t *in = heap_caps_malloc(sizeof(*in), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    ps_detections_t *dets = heap_caps_malloc(sizeof(*dets), MALLOC_CAP_8BIT);
    uint32_t t_last_run = 0, bench_until = 0;
    int overruns = 0;
    esp_task_wdt_add(NULL);
    for (;;) {
        esp_task_wdt_reset();
        uint32_t frame_id;
        if (xTaskNotifyWait(0, 0, &frame_id, pdMS_TO_TICKS(1000)) != pdTRUE) continue;

        const uint32_t now = app_now_ms();
        app_lock();
        const uint8_t max_fps = g_app.sys->policy.max_infer_fps;
        const uint32_t frame_t = g_app.sys->frame.t_ms;
        const float thr = g_app.sys->cfg.score_threshold;
        memcpy(in, g_app.infer_in, sizeof(*in));   /* snapshot of the latest frame */
        app_unlock();

        /* Power-policy rate cap (the Lepton gives ~8.7 fps; NORMAL allows 16). */
        if (max_fps && now - t_last_run < 1000u / max_fps - 5) {
            g_app.infer_skipped_fps_cap++;
            continue;
        }
        t_last_run = now;

        bool use_neural = g_app.neural_ok && (int32_t)(now - bench_until) >= 0;
        bool neural_done = false;
        power_busy_begin();
        if (use_neural) {
            esp_err_t e = detector_run(in, thr, CONFIG_PS_INFER_DEADLINE_MS * 1000u, dets);
            if (e == ESP_OK) {
                overruns = 0;
                neural_done = true;
            } else {
                g_app.infer_fallbacks++;
                if (e == ESP_ERR_TIMEOUT && ++overruns >= NEURAL_MAX_CONSECUTIVE_OVERRUNS) {
                    ESP_LOGW(TAG, "inference overran %d times (last %lu us): classical for %d s", overruns,
                             (unsigned long)dets->latency_us, NEURAL_BENCH_MS / 1000);
                    bench_until = now + NEURAL_BENCH_MS;
                    overruns = 0;
                }
            }
            if (dets->latency_us > g_app.infer_max_us) g_app.infer_max_us = dets->latency_us;
        }

        app_lock();
        if (neural_done) {
            dets->frame_id = frame_id;
            dets->t_ms = frame_t;
            g_app.sys->stats.last_infer_us = dets->latency_us;
            ps_system_on_detections(g_app.sys, dets, app_now_ms());
            g_app.infer_neural++;
        } else {
            /* Classical fallback on the system's own (latest) frame. */
            const int64_t t0 = esp_timer_get_time();
            ps_system_run_classical(g_app.sys);
            g_app.sys->stats.last_infer_us = us_since(t0);
            g_app.infer_classical++;
        }
        app_unlock();
        power_busy_end();
    }
}

/* ------------------------------------------------------------------------ */
/* IMU: INT-driven, core 0.                                                   */
/* ------------------------------------------------------------------------ */

static void imu_task(void *arg)
{
    uint16_t rate = 0;
    esp_task_wdt_add(NULL);
    for (;;) {
        esp_task_wdt_reset();
        bno085_service(g_app.imu, 50);   /* callbacks take the lock per event */
        app_lock();
        uint16_t want = g_app.sys->policy.imu_rate_hz;
        app_unlock();
        if (want != rate) {
            if (bno085_set_rv_rate(g_app.imu, want) == ESP_OK) rate = want;
        }
    }
}

/* ------------------------------------------------------------------------ */
/* 100 Hz tick: health supervisor, navigation, alert generation.            */
/* ------------------------------------------------------------------------ */

static void tick_task(void *arg)
{
    TickType_t last = xTaskGetTickCount();
    esp_task_wdt_add(NULL);
    for (;;) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(10));
        esp_task_wdt_reset();
        app_lock();
        ps_system_tick(g_app.sys, app_now_ms());
        app_unlock();
    }
}

/* ------------------------------------------------------------------------ */
/* Display: render under the lock, PPA scale + DSI flip outside it.          */
/* ------------------------------------------------------------------------ */

static void display_task(void *arg)
{
    TickType_t last = xTaskGetTickCount();
    uint8_t brightness_sent = 0;
    esp_task_wdt_add(NULL);
    for (;;) {
        esp_task_wdt_reset();
        power_busy_begin();
        app_lock();
        ps_system_render(g_app.sys, g_app.fb, app_now_ms());
        const uint8_t br = ps_system_brightness(g_app.sys);
        uint8_t fps = g_app.sys->policy.display_fps;
        app_unlock();
        microoled_present(g_app.fb);
        power_busy_end();
        if (br != brightness_sent && microoled_set_brightness(br) == ESP_OK) brightness_sent = br;
        if (fps < 5) fps = 5;
        vTaskDelayUntil(&last, pdMS_TO_TICKS(1000 / fps));
    }
}

/* ------------------------------------------------------------------------ */
/* Audio: phrase sequences, interrupted by more urgent alerts.               */
/* ------------------------------------------------------------------------ */

static bool audio_should_abort(void *ctx, int playing_prio)
{
    esp_task_wdt_reset();   /* called every 32 ms of audio */
    app_lock();
    int top = ps_alerts_peek_max_prio(&g_app.sys->alerts);
    app_unlock();
    return top > playing_prio;
}

static void audio_task(void *arg)
{
    esp_task_wdt_add(NULL);
    for (;;) {
        esp_task_wdt_reset();
        ps_alert_t a;
        app_lock();
        bool got = ps_system_next_alert(g_app.sys, &a);
        app_unlock();
        if (!got) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }
        if (g_app.audio_ok) audio_play_alert(&a, audio_should_abort, NULL);
    }
}

/* ------------------------------------------------------------------------ */
/* Service: buttons, battery, power policy, statistics.                      */
/* ------------------------------------------------------------------------ */

static void log_stats(uint32_t now, uint32_t dt_ms, uint32_t *last_frames)
{
    ps_stats_t s;
    uint8_t det_src, pct;
    ps_power_mode_t mode;
    bool cam, imu;
    app_lock();
    s = g_app.sys->stats;
    det_src = (uint8_t)g_app.sys->dets.source;
    pct = g_app.sys->power.pct;
    mode = g_app.sys->power.mode;
    cam = g_app.sys->camera_ok;
    imu = g_app.sys->imu_ok;
    app_unlock();

    const float fps = dt_ms ? (s.frames - *last_frames) * 1000.0f / dt_ms : 0;
    *last_frames = s.frames;
    ESP_LOGI(TAG, "frames %lu (%.1f fps, dropped %lu) pipe %lu/%lu us | det %s infer %lu/%lu us "
             "nn %lu cl %lu fb %lu capskip %lu | cam %d imu %d | batt %u mV %u%% mode %d cpu %u MHz | lockwait %lu us",
             (unsigned long)s.frames, fps, (unsigned long)s.frames_dropped,
             (unsigned long)s.last_pipeline_us, (unsigned long)g_app.pipeline_max_us,
             det_src == PS_DETECTOR_NEURAL ? "NN" : det_src == PS_DETECTOR_CLASSICAL ? "CL" : "--",
             (unsigned long)s.last_infer_us, (unsigned long)g_app.infer_max_us,
             (unsigned long)g_app.infer_neural, (unsigned long)g_app.infer_classical,
             (unsigned long)g_app.infer_fallbacks, (unsigned long)g_app.infer_skipped_fps_cap,
             cam, imu, g_app.batt_mv, pct, (int)mode, power_cpu_mhz(),
             (unsigned long)g_app.lock_wait_max_us);

    if (g_app.lepton) {
        lepton_stats_t l;
        lepton_get_stats(g_app.lepton, &l);
        ESP_LOGI(TAG, "lepton: pkts %lu disc %lu crc %lu seq %lu seg %lu inval %lu order %lu frames %lu lost %lu "
                 "resync %lu stall %lu reboot %lu vsync %lu/%lu to, spi err %lu, %lu us",
                 (unsigned long)l.vospi.packets, (unsigned long)l.vospi.discards, (unsigned long)l.vospi.crc_errors,
                 (unsigned long)l.vospi.seq_errors, (unsigned long)l.vospi.segments, (unsigned long)l.vospi.seg_invalid,
                 (unsigned long)l.vospi.seg_order_errors, (unsigned long)l.vospi.frames,
                 (unsigned long)l.vospi.frames_lost, (unsigned long)l.vospi.resyncs, (unsigned long)l.stall_resyncs,
                 (unsigned long)l.reboots, (unsigned long)l.vsync_irqs, (unsigned long)l.vsync_timeouts,
                 (unsigned long)l.spi_errors, (unsigned long)l.last_capture_us);
    }
    if (g_app.imu) {
        bno085_stats_t b;
        bno085_get_stats(g_app.imu, &b);
        ESP_LOGI(TAG, "imu: rv %lu acc %lu steps %lu | pkts %lu unk %lu trunc %lu seqgap %lu resets %lu reinit %lu i2c err %lu",
                 (unsigned long)b.rv_reports, (unsigned long)b.accel_reports, (unsigned long)b.steps,
                 (unsigned long)b.parser.packets, (unsigned long)b.parser.unknown, (unsigned long)b.parser.truncated,
                 (unsigned long)b.parser.seq_gaps, (unsigned long)b.parser.resets, (unsigned long)b.reinits,
                 (unsigned long)b.i2c_errors);
    }
    if (g_app.neural_ok) {
        detector_stats_t d;
        detector_get_stats(&d);
        ESP_LOGI(TAG, "nn: runs %lu avg %.1f ms max %.1f ms overruns %lu", (unsigned long)d.runs, d.avg_us / 1000.0f,
                 d.max_us / 1000.0f, (unsigned long)d.overruns);
    }
    if (g_app.display_ok) {
        microoled_stats_t m;
        microoled_get_stats(&m);
        ESP_LOGI(TAG, "display: frames %lu ppa %lu us present %lu us fliptimeouts %lu ppaerr %lu",
                 (unsigned long)m.frames, (unsigned long)m.last_ppa_us, (unsigned long)m.last_present_us,
                 (unsigned long)m.flip_timeouts, (unsigned long)m.ppa_errors);
    }
    if (g_app.audio_ok) {
        audio_stats_t a;
        audio_get_stats(&a);
        ESP_LOGI(TAG, "audio: alerts %lu clips %lu beeps %lu missing %lu aborted %lu pack %s (%u clips)",
                 (unsigned long)a.alerts, (unsigned long)a.clips, (unsigned long)a.beeps,
                 (unsigned long)a.missing_clips, (unsigned long)a.aborted, a.pack_ok ? "ok" : "MISSING",
                 a.clips_available);
    }
    ESP_LOGI(TAG, "heap: internal %u B (min %u), PSRAM %u B", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    g_app.pipeline_max_us = g_app.infer_max_us = g_app.lock_wait_max_us = 0;
}

static void service_task(void *arg)
{
    TickType_t last = xTaskGetTickCount();
    uint32_t t_batt = 0, t_stats = app_now_ms(), frames_at_stats = 0;
    esp_task_wdt_add(NULL);
    for (;;) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(10));
        esp_task_wdt_reset();
        const uint32_t now = app_now_ms();

        ps_button_t ev[2];
        int n = buttons_poll(now, ev);
        if (n) {
            app_lock();
            for (int i = 0; i < n; i++) ps_system_on_button(g_app.sys, ev[i], now);
            app_unlock();
            g_app.buttons_pressed += n;
        }

        if (now - t_batt >= 1000) {
            t_batt = now;
            uint16_t mv = power_read_batt_mv();
            g_app.batt_mv = mv;
            ps_power_policy_t pol;
            app_lock();
            if (mv) ps_system_on_battery(g_app.sys, mv, now);
            pol = g_app.sys->policy;
            app_unlock();
            power_apply_policy(&pol);   /* CPU ceiling; tasks read the rest themselves */
        }

        if (now - t_stats >= CONFIG_PS_STATS_PERIOD_MS) {
            log_stats(now, now - t_stats, &frames_at_stats);
            t_stats = now;
        }
    }
}

/* ------------------------------------------------------------------------ */

void app_start_tasks(void)
{
    /* Inference first so the thermal task always has someone to notify. */
    xTaskCreatePinnedToCore(inference_task, "infer", 16384, NULL, 15, &g_app.infer_task, 1);
    if (g_app.lepton) xTaskCreatePinnedToCore(thermal_task, "thermal", 6144, NULL, 20, NULL, 0);
    if (g_app.imu) xTaskCreatePinnedToCore(imu_task, "imu", 4096, NULL, 18, NULL, 0);
    xTaskCreatePinnedToCore(tick_task, "tick", 4096, NULL, 16, NULL, 0);
    if (g_app.display_ok) xTaskCreatePinnedToCore(display_task, "display", 6144, NULL, 10, NULL, 1);
    xTaskCreatePinnedToCore(audio_task, "audio", 4096, NULL, 8, NULL, 0);
    xTaskCreatePinnedToCore(service_task, "service", 4096, NULL, 3, NULL, 0);
}
