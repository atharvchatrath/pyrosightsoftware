# Bosch BNO085 IMU over I2C (SHTP / SH-2). shtp_proto.c is host-tested.
idf_component_register(
    SRCS "bno085.c" "shtp_proto.c"
    INCLUDE_DIRS "include"
    REQUIRES pyrosight_core esp_driver_i2c esp_driver_gpio esp_timer
)
