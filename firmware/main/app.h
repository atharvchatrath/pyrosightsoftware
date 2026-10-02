/*
 * PyroSight application: shared state and task entry points.
 *
 * Task / core layout (priorities: higher number = more urgent):
 *
 *   task        core prio  period / trigger        work
 *   thermal      0    20   Lepton VSYNC (~106 Hz   VoSPI capture outside the lock; ps_system_on_frame
 *                          segments, ~8.7 fps)     under the lock; hands model_input to inference
 *   imu          0    18   BNO085 INT (100 Hz)     SHTP read outside the lock; on_yaw/step/accel under it
 *   tick         0    16   10 ms                   ps_system_tick (health, nav confidence, alerts)
 *   inference    1    15   new frame               ESP-DL on a private copy, no lock held;
 *                                                  ps_system_on_detections / run_classical under it
 *   display      1    10   1/display_fps           ps_system_render under the lock; PPA + DSI flip outside
 *   audio        0     8   alert queue (50 ms)     ps_system_next_alert under the lock; playback outside
 *   service      0     3   10 ms                   buttons (10 ms), battery + power policy (1 s), stats (5 s)
 *
 * The one mutex (g_app.lock) serialises every ps_system_* call; each critical
 * section is short (<= ~3 ms for on_frame at 360 MHz, see README timing budget).
 */
#ifndef APP_H
#define APP_H

#include <stdbool.h>
#include <stdint.h>

#include "bno085.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lepton.h"
#include "pyrosight/ps_system.h"

typedef struct {
    ps_system_t *sys;           /* PSRAM */
    SemaphoreHandle_t lock;

    lepton_dev_t *lepton;
    bno085_t *imu;
    bool display_ok, audio_ok, neural_ok;

    /* Inference hand-off: written by the thermal task under the lock. */
    ps_gray_frame_t *infer_in;  /* private copy owned by the inference task */
    TaskHandle_t infer_task;

    ps_fb_t *fb;                /* render target, PSRAM, DMA/PPA reachable */

    /* Statistics not kept by ps_system. */
    volatile uint32_t infer_skipped_fps_cap;
    volatile uint32_t infer_neural, infer_classical, infer_fallbacks;
    volatile uint32_t pipeline_max_us, infer_max_us;
    volatile uint32_t lock_wait_max_us;
    volatile uint32_t buttons_pressed;
    volatile uint16_t batt_mv;
} app_t;

extern app_t g_app;

/* Lock helpers (record the worst wait time). */
void app_lock(void);
void app_unlock(void);
uint32_t app_now_ms(void);

void app_start_tasks(void);

#endif /* APP_H */
