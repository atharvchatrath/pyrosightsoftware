# Voice prompts: ES8311 codec (via espressif/esp_codec_dev) + I2S, clips from
# the "audio" flash partition. audio_pack.c is host-tested.
idf_component_register(
    SRCS "audio.c" "audio_pack.c"
    INCLUDE_DIRS "include"
    REQUIRES pyrosight_core esp_driver_i2s esp_driver_i2c esp_driver_gpio esp_partition
)
