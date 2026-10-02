/*
 * Board pin map: ESP32-P4-Function-EV-Board + PyroSight sensors.
 * Every value comes from Kconfig (main/Kconfig.projbuild, menuconfig ->
 * PyroSight); this header only gives them short names and fixes the
 * peripheral instances.
 *
 *   Function            Signal         Default GPIO   Notes
 *   Lepton VoSPI        SCLK/MISO/CS   20 / 21 / 22   SPI2, mode 3, <= 20 MHz
 *   Lepton VSYNC        GPIO3 out      23             rising edge per segment
 *   Lepton CCI          I2C 0x2A       sensor bus
 *   BNO085              I2C 0x4A       sensor bus     INT 47, RST 48
 *   Sensor I2C          SDA / SCL      32 / 33        I2C1 (separate from codec)
 *   ES8311 codec        I2C 0x18       7 / 8          I2C0 (board)
 *   I2S                 MCLK/BCLK/WS/DOUT 13/12/10/9  I2S0 (board)
 *   Speaker PA enable                  53             board NS4150; -1 for jack only
 *   Micro-OLED          MIPI-DSI       dedicated      2 lanes, reset 27
 *   Button A / B                       45 / 46        to GND, internal pull-up
 *   Battery ADC         divider        16             ADC1
 */
#ifndef BOARD_PINS_H
#define BOARD_PINS_H

#include "sdkconfig.h"

#define BOARD_LEPTON_SPI_HOST   SPI2_HOST
#define BOARD_LEPTON_SCLK       CONFIG_PS_LEPTON_SPI_SCLK
#define BOARD_LEPTON_MISO       CONFIG_PS_LEPTON_SPI_MISO
#define BOARD_LEPTON_CS         CONFIG_PS_LEPTON_SPI_CS
#define BOARD_LEPTON_VSYNC      CONFIG_PS_LEPTON_VSYNC
#define BOARD_LEPTON_RESET      CONFIG_PS_LEPTON_RESET
#define BOARD_LEPTON_PWR_DN     CONFIG_PS_LEPTON_PWR_DN

#define BOARD_CODEC_I2C_PORT    I2C_NUM_0
#define BOARD_CODEC_I2C_SDA     CONFIG_PS_CODEC_I2C_SDA
#define BOARD_CODEC_I2C_SCL     CONFIG_PS_CODEC_I2C_SCL

#if CONFIG_PS_SENSORS_ON_CODEC_BUS
#define BOARD_SENSOR_I2C_SEPARATE 0
#else
#define BOARD_SENSOR_I2C_SEPARATE 1
#define BOARD_SENSOR_I2C_PORT   I2C_NUM_1
#define BOARD_SENSOR_I2C_SDA    CONFIG_PS_SENSOR_I2C_SDA
#define BOARD_SENSOR_I2C_SCL    CONFIG_PS_SENSOR_I2C_SCL
#endif

#define BOARD_BNO085_INT        CONFIG_PS_BNO085_INT
#define BOARD_BNO085_RST        CONFIG_PS_BNO085_RST

#define BOARD_I2S_PORT          0
#define BOARD_I2S_MCLK          CONFIG_PS_I2S_MCLK
#define BOARD_I2S_BCLK          CONFIG_PS_I2S_BCLK
#define BOARD_I2S_WS            CONFIG_PS_I2S_WS
#define BOARD_I2S_DOUT          CONFIG_PS_I2S_DOUT
#define BOARD_PA_ENABLE         CONFIG_PS_PA_ENABLE

#define BOARD_OLED_RESET        CONFIG_PS_OLED_RESET

#define BOARD_BUTTON_A          CONFIG_PS_BUTTON_A
#define BOARD_BUTTON_B          CONFIG_PS_BUTTON_B
#define BOARD_BATT_ADC_GPIO     CONFIG_PS_BATT_ADC_GPIO

/* Kconfig booleans as 0/1 values. */
#ifdef CONFIG_PS_LIGHT_SLEEP
#define PS_CFG_LIGHT_SLEEP 1
#else
#define PS_CFG_LIGHT_SLEEP 0
#endif
#ifdef CONFIG_PS_OLED_BRIGHTNESS_2BYTES
#define PS_CFG_OLED_BRIGHTNESS_2BYTES 1
#else
#define PS_CFG_OLED_BRIGHTNESS_2BYTES 0
#endif
#ifdef CONFIG_PS_OLED_RGB888
#define PS_CFG_OLED_RGB888 1
#else
#define PS_CFG_OLED_RGB888 0
#endif
#ifdef CONFIG_PS_LEPTON_CHECK_CRC
#define PS_CFG_LEPTON_CHECK_CRC 1
#else
#define PS_CFG_LEPTON_CHECK_CRC 0
#endif

#endif /* BOARD_PINS_H */
