/*
 * BNO085 driver (ESP-IDF, new I2C master driver, INT-driven).
 *
 * Enables Game Rotation Vector (0x08, Q14), Step Detector (0x18) and Linear
 * Acceleration (0x04, Q8) with SH-2 Set Feature. Re-enables them after any
 * hub reset (reset-complete on channel 1) and resets the hub itself if it
 * goes silent. Events are delivered through callbacks with esp_timer-based
 * millisecond timestamps corrected by the SH-2 report delay.
 */
#ifndef BNO085_H
#define BNO085_H

#include <stdbool.h>
#include <stdint.h>

#include "driver/i2c_master.h"
#include "esp_err.h"
#include "shtp_proto.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    i2c_master_bus_handle_t bus;
    uint8_t addr;           /* 0x4A (SA0 low) or 0x4B */
    uint32_t i2c_hz;        /* <= 400 kHz */
    int pin_int;            /* H_INTN, active low, required */
    int pin_rst;            /* NRST, active low; -1: soft reset only */
    uint16_t rv_rate_hz;    /* game rotation vector rate, e.g. 100 */
    uint16_t accel_rate_hz; /* linear acceleration rate, e.g. 50 */
    uint32_t silence_ms;    /* no report for this long -> reset + re-init (default 1000) */
} bno085_config_t;

typedef struct {
    void (*on_yaw)(void *ctx, float yaw_rad, uint32_t t_ms);
    void (*on_step)(void *ctx, uint32_t t_ms);
    void (*on_linear_accel)(void *ctx, float mag_mps2, uint32_t t_ms);
    void *ctx;
} bno085_handlers_t;

typedef struct {
    sh2_parser_t parser;    /* packets, reports, unknown, truncated, resets, seq gaps */
    uint32_t int_irqs;
    uint32_t i2c_errors;
    uint32_t reinits;       /* silence-triggered resets */
    uint32_t configs;       /* feature enables sent */
    uint32_t rv_reports, steps, accel_reports;
    uint8_t sw_major, sw_minor;
    uint32_t sw_build;
} bno085_stats_t;

typedef struct bno085_dev bno085_t;

esp_err_t bno085_init(const bno085_config_t *cfg, const bno085_handlers_t *h, bno085_t **out);

/* Wait up to timeout_ms for INT, then drain and dispatch pending packets.
 * Also runs the silence watchdog. Call in a loop from the IMU task. */
esp_err_t bno085_service(bno085_t *d, uint32_t timeout_ms);

/* Change the rotation-vector rate (power policy). Takes effect immediately. */
esp_err_t bno085_set_rv_rate(bno085_t *d, uint16_t hz);

void bno085_get_stats(bno085_t *d, bno085_stats_t *out);

#ifdef __cplusplus
}
#endif

#endif /* BNO085_H */
