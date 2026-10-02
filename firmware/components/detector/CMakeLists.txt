# Neural fire/person detector (ESP-DL v3, C++), exposed through a C API.
# detector_quant.c is host-tested.
idf_component_register(
    SRCS "detector.cpp" "detector_quant.c"
    INCLUDE_DIRS "include"
    REQUIRES pyrosight_core esp_partition esp_timer
)
