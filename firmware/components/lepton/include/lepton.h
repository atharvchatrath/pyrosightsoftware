/*
 * FLIR Lepton 3.5 driver for ESP-IDF (ESP32-P4).
 *
 *   - CCI over the new I2C master driver (address 0x2A), see lepton_cci.h;
 *   - VoSPI over SPI mode 3 (<= 20 MHz), RX only (MOSI unused);
 *   - VSYNC on Lepton GPIO3 (enabled through CCI) -> ESP32 GPIO interrupt:
 *     one pulse per segment (~106 Hz on Lepton 3.x), the segment can be read
 *     right after the pulse. Without VSYNC (pin -1 or no pulses) the driver
 *     polls instead and still works, at a higher bus load.
 *
 * lepton_capture() blocks until a complete radiometric frame (4 segments,
 * ~8.7 unique frames/s on the export-compliant Lepton 3.5) is assembled and
 * converted to deci-Celsius. Loss of sync is handled internally (CS idled
 * > 185 ms, counters in lepton_stats_t).
 */
#ifndef LEPTON_H
#define LEPTON_H

#include <stdbool.h>
#include <stdint.h>

#include "driver/i2c_master.h"
#include "driver/spi_master.h"
#include "esp_err.h"
#include "lepton_cci.h"
#include "lepton_vospi_proto.h"
#include "pyrosight/ps_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    spi_host_device_t spi_host;
    int pin_sclk, pin_miso, pin_cs;
    int pin_vsync;          /* -1: poll */
    int pin_reset;          /* -1: not wired (active low) */
    int pin_pwr_dn;         /* -1: not wired (active low) */
    int spi_hz;             /* <= 20 MHz */
    i2c_master_bus_handle_t i2c_bus;
    uint32_t i2c_hz;        /* <= 1 MHz, 400 kHz recommended */
    bool check_crc;
    /* Called during long internal waits (resync, camera reboot: up to ~7 s)
     * so the capture task can feed its task watchdog. May be NULL. */
    void (*keepalive)(void);
} lepton_config_t;

typedef struct {
    lep_vospi_counters_t vospi;
    uint32_t vsync_irqs;
    uint32_t vsync_timeouts;   /* waited for VSYNC and read anyway */
    uint32_t spi_errors;
    uint32_t stall_resyncs;    /* no frame for LEPTON_STALL_MS */
    uint32_t reboots;          /* CCI reboot after repeated stalls */
    uint32_t last_capture_us;  /* SPI time of the last frame */
    lep_setup_report_t setup;
    bool setup_ok;
} lepton_stats_t;

typedef struct lepton_dev lepton_dev_t;

esp_err_t lepton_init(const lepton_config_t *cfg, lepton_dev_t **out);

/*
 * Block until a new frame is ready (or timeout_ms passes: ESP_ERR_TIMEOUT).
 * frame_id and t_ms (esp_timer, ms) are filled in.
 */
esp_err_t lepton_capture(lepton_dev_t *d, ps_thermal_frame_t *out, uint32_t timeout_ms);

/* Flat-field correction (shutter). Call from the capture task, not concurrently with capture. */
esp_err_t lepton_run_ffc(lepton_dev_t *d);
esp_err_t lepton_get_status(lepton_dev_t *d, lep_cam_status_t *st);
esp_err_t lepton_get_fpa_temp_dc(lepton_dev_t *d, int16_t *dc);

void lepton_get_stats(lepton_dev_t *d, lepton_stats_t *out);

#ifdef __cplusplus
}
#endif

#endif /* LEPTON_H */
