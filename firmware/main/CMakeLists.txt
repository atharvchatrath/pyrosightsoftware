idf_component_register(
    SRCS "main.c" "app_tasks.c"
    INCLUDE_DIRS "."
    REQUIRES pyrosight_core lepton bno085 microoled audio detector power buttons
             esp_driver_i2c esp_driver_spi esp_timer esp_system freertos
)
