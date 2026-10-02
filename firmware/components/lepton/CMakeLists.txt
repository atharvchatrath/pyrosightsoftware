# FLIR Lepton 3.5 driver: VoSPI capture (SPI + VSYNC) and CCI (I2C).
# lepton_vospi_proto.c and lepton_cci.c are hardware independent and are also
# built by firmware/test_host/ on the host.
idf_component_register(
    SRCS "lepton.c" "lepton_vospi_proto.c" "lepton_cci.c"
    INCLUDE_DIRS "include"
    REQUIRES pyrosight_core esp_driver_spi esp_driver_i2c esp_driver_gpio esp_timer
)
