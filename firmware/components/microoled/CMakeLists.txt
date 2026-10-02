# 0.39" micro-OLED over MIPI-DSI (esp_lcd DPI video mode) with PPA scaling.
idf_component_register(
    SRCS "microoled.c"
    INCLUDE_DIRS "include"
    REQUIRES pyrosight_core esp_lcd esp_driver_ppa esp_driver_gpio esp_hw_support esp_mm
)
