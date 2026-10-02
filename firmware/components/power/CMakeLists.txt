# Battery ADC and power-policy application. power_calc.c is host-tested.
idf_component_register(
    SRCS "power.c" "power_calc.c"
    INCLUDE_DIRS "include"
    REQUIRES pyrosight_core esp_adc esp_pm
)
