/*
 * FLIR Lepton CCI (Camera Control Interface) over I2C, address 0x2A.
 *
 * Register map (Lepton Software IDD, "CCI register interface"):
 *   0x0000 Power-on register
 *   0x0002 STATUS     bit0 BUSY, bit1 boot mode (1 = normal), bit2 boot status
 *                     (1 = booted), bits 15:8 last command result (int8 LEP_RESULT)
 *   0x0004 COMMAND ID
 *   0x0006 DATA LENGTH (number of 16-bit words)
 *   0x0008..0x0026 DATA 0..15
 *   0xF800 block data buffer 0 (transfers > 16 words; not used here)
 * Registers are 16-bit big endian; the register address is sent first (16 bit).
 *
 * Command ID word: [15:14] reserved, bit 14 = OEM/RAD protection bit,
 * [11:8] module ID, [7:2] command base, [1:0] type (0 GET, 1 SET, 2 RUN).
 *
 * Sequence: wait !BUSY -> (SET: write DATA n) -> write DATA LENGTH ->
 * write COMMAND ID -> wait !BUSY -> check STATUS[15:8] == 0 -> (GET: read DATA n).
 *
 * 32-bit values (enums, uint32) occupy two data words, least significant
 * word first (DATA0 = LSW). This matches FLIR's LeptonSDK (LEP_I2C_*), but
 * verify on hardware with a GET after SET; lep_cci_set_u32_verified() does it.
 *
 * This file is transport-agnostic: the caller supplies register access
 * callbacks, so the sequencing is unit-tested on the host with a fake sensor.
 */
#ifndef LEPTON_CCI_H
#define LEPTON_CCI_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LEP_CCI_I2C_ADDR 0x2A

#define LEP_REG_POWER   0x0000
#define LEP_REG_STATUS  0x0002
#define LEP_REG_COMMAND 0x0004
#define LEP_REG_DATALEN 0x0006
#define LEP_REG_DATA0   0x0008

#define LEP_STATUS_BUSY        0x0001
#define LEP_STATUS_BOOT_MODE   0x0002
#define LEP_STATUS_BOOT_STATUS 0x0004

#define LEP_TYPE_GET 0x0
#define LEP_TYPE_SET 0x1
#define LEP_TYPE_RUN 0x2

#define LEP_PROT_BIT 0x4000  /* required for OEM (0x08) and RAD (0x0E) modules */

/* Module IDs (bits 11:8). */
#define LEP_MOD_AGC 0x0100
#define LEP_MOD_SYS 0x0200
#define LEP_MOD_VID 0x0300
#define LEP_MOD_OEM (0x0800 | LEP_PROT_BIT)
#define LEP_MOD_RAD (0x0E00 | LEP_PROT_BIT)

/*
 * Command bases (module | base); add LEP_TYPE_* to issue.
 * Confidence: the ones marked [sure] are in FLIR's published LeptonSDK
 * headers and widely used examples; [check] are from memory of the IDD
 * (rev 200+) and should be confirmed against the IDD for your firmware.
 */
#define LEP_CID_AGC_ENABLE          (LEP_MOD_AGC | 0x00)  /* [sure]  u32 enum 0 off / 1 on */
#define LEP_CID_SYS_PING            (LEP_MOD_SYS | 0x00)  /* [sure]  RUN */
#define LEP_CID_SYS_STATUS          (LEP_MOD_SYS | 0x04)  /* [sure]  GET: camStatus u32, cmdCount u16, reserved u16 */
#define LEP_CID_SYS_SERIAL          (LEP_MOD_SYS | 0x08)  /* [sure]  GET 4 words */
#define LEP_CID_SYS_UPTIME          (LEP_MOD_SYS | 0x0C)  /* [sure]  GET u32 ms */
#define LEP_CID_SYS_FPA_TEMP_K      (LEP_MOD_SYS | 0x14)  /* [sure]  GET u16 Kelvin x 100 */
#define LEP_CID_SYS_TELEMETRY_EN    (LEP_MOD_SYS | 0x18)  /* [sure]  u32 enum */
#define LEP_CID_SYS_FFC_SHUTTER_MODE (LEP_MOD_SYS | 0x3C) /* [check] struct, not used */
#define LEP_CID_SYS_RUN_FFC         (LEP_MOD_SYS | 0x40)  /* [sure]  RUN (0x0242) */
#define LEP_CID_SYS_FFC_STATUS      (LEP_MOD_SYS | 0x44)  /* [check] GET u32 enum: 0 ready, 1 busy(?), see IDD */
#define LEP_CID_SYS_GAIN_MODE       (LEP_MOD_SYS | 0x48)  /* [check] u32 enum 0 HIGH, 1 LOW, 2 AUTO */
#define LEP_CID_OEM_REBOOT          (LEP_MOD_OEM | 0x40)  /* [check] RUN (0x4842) */
#define LEP_CID_OEM_GPIO_MODE       (LEP_MOD_OEM | 0x54)  /* [sure]  u32 enum, 5 = VSYNC */
#define LEP_CID_RAD_ENABLE          (LEP_MOD_RAD | 0x10)  /* [sure]  u32 enum */
#define LEP_CID_RAD_TLINEAR_EN      (LEP_MOD_RAD | 0xC0)  /* [sure]  u32 enum */
#define LEP_CID_RAD_TLINEAR_RES     (LEP_MOD_RAD | 0xC4)  /* [sure]  u32 enum 0 = 0.1 K, 1 = 0.01 K */
#define LEP_CID_RAD_TLINEAR_AUTORES (LEP_MOD_RAD | 0xC8)  /* [check] u32 enum, keep disabled */

#define LEP_GPIO_MODE_VSYNC 5
#define LEP_GAIN_HIGH 0
#define LEP_GAIN_LOW  1
#define LEP_GAIN_AUTO 2

/* Camera status (LEP_SYS_CAM_STATUS_STATES). */
typedef enum {
    LEP_CAM_READY = 0,
    LEP_CAM_INITIALIZING = 1,
    LEP_CAM_LOW_POWER = 2,
    LEP_CAM_GOING_STANDBY = 3,
    LEP_CAM_FLAT_FIELD = 4,  /* FFC in progress */
} lep_cam_status_t;

typedef struct {
    /* Register access; return 0 on success. */
    int (*read_reg)(void *ctx, uint16_t reg, uint16_t *val);
    int (*write_reg)(void *ctx, uint16_t reg, uint16_t val);
    void (*delay_ms)(void *ctx, uint32_t ms);
    void *ctx;
    uint32_t busy_timeout_ms;  /* default 1000 (FFC can take ~0.5 s) */
} lep_cci_t;

typedef enum {
    LEP_CCI_OK = 0,
    LEP_CCI_ERR_IO = -1,
    LEP_CCI_ERR_TIMEOUT = -2,
    LEP_CCI_ERR_CAMERA = -3,     /* STATUS[15:8] non-zero; see last_result */
    LEP_CCI_ERR_VERIFY = -4,
} lep_cci_err_t;

typedef struct {
    int8_t last_result;          /* LEP_RESULT from STATUS[15:8] */
    uint32_t commands, errors, timeouts;
} lep_cci_stats_t;

extern lep_cci_stats_t lep_cci_stats;

static inline uint16_t lep_cci_cmd(uint16_t base, uint16_t type) { return (uint16_t)(base | type); }

int lep_cci_wait_ready(const lep_cci_t *c);
/* Wait for STATUS boot bits (normal mode, booted); Lepton needs ~1-2 s from power-up. */
int lep_cci_wait_boot(const lep_cci_t *c, uint32_t timeout_ms);

int lep_cci_get(const lep_cci_t *c, uint16_t base, uint16_t *words, uint16_t n);
int lep_cci_set(const lep_cci_t *c, uint16_t base, const uint16_t *words, uint16_t n);
int lep_cci_run(const lep_cci_t *c, uint16_t base);

int lep_cci_get_u32(const lep_cci_t *c, uint16_t base, uint32_t *v);
int lep_cci_set_u32(const lep_cci_t *c, uint16_t base, uint32_t v);
/* SET then GET and compare. */
int lep_cci_set_u32_verified(const lep_cci_t *c, uint16_t base, uint32_t v);

typedef struct {
    bool agc_off;
    bool radiometry_on;
    bool tlinear_on;
    bool low_gain;
    bool res_0_1k;
    bool telemetry_off;
    bool vsync_on;
} lep_setup_report_t;

/*
 * PyroSight configuration: AGC off, radiometry on, TLinear on, low gain
 * (fire up to ~450 C), TLinear resolution 0.1 K, telemetry off, GPIO3 = VSYNC.
 * Continues past individual failures and reports what stuck. Returns 0 if
 * everything succeeded.
 */
int lep_cci_setup_pyrosight(const lep_cci_t *c, lep_setup_report_t *rep);

int lep_cci_run_ffc(const lep_cci_t *c);
int lep_cci_get_status(const lep_cci_t *c, lep_cam_status_t *st, uint16_t *cmd_count);
int lep_cci_get_fpa_temp_dc(const lep_cci_t *c, int16_t *dc);

#ifdef __cplusplus
}
#endif

#endif /* LEPTON_CCI_H */
