/*
 * PyroSight core: shared types.
 *
 * The core library is portable C99 with no heap allocation and no platform
 * dependencies. It is compiled into the ESP32-P4 firmware and into the host
 * simulator / unit tests, so every algorithm runs identically in both places.
 */
#ifndef PS_TYPES_H
#define PS_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* FLIR Lepton 3.5 geometry. */
#define PS_THERM_W 160
#define PS_THERM_H 120
#define PS_THERM_PIXELS (PS_THERM_W * PS_THERM_H)

/*
 * Temperatures inside the core are int16 in deci-degrees Celsius
 * (e.g. 365 == 36.5 C). Range -3276.8 .. +3276.7 C covers the Lepton 3.5
 * low-gain range (up to ~450 C, ~600 C with extended calibration).
 * The Lepton driver converts TLinear Kelvin units into this representation.
 */
typedef int16_t ps_dc_t;

#define PS_DC(celsius) ((ps_dc_t)((celsius) * 10))

/* A radiometric thermal frame. */
typedef struct {
    ps_dc_t px[PS_THERM_PIXELS];
    uint32_t frame_id;  /* monotonically increasing */
    uint32_t t_ms;      /* capture timestamp, milliseconds */
} ps_thermal_frame_t;

/* An 8-bit grayscale image at sensor resolution (display or model input). */
typedef struct {
    uint8_t px[PS_THERM_PIXELS];
} ps_gray_frame_t;

typedef struct {
    float x, y;
} ps_vec2_t;

#ifdef __cplusplus
}
#endif

#endif /* PS_TYPES_H */
