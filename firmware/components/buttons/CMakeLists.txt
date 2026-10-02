# Two-button input. button_fsm.c is host-tested.
idf_component_register(
    SRCS "buttons.c" "button_fsm.c"
    INCLUDE_DIRS "include"
    REQUIRES pyrosight_core esp_driver_gpio
)
