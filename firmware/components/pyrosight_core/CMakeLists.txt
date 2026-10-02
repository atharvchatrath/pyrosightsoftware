# The portable PyroSight core (core/), compiled unchanged into the firmware.
# Sources are referenced in place, never copied, so the device runs exactly
# the code the host tests and the simulator run.
set(PS_CORE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../core")
file(GLOB PS_CORE_SRCS "${PS_CORE_DIR}/src/*.c")

idf_component_register(
    SRCS ${PS_CORE_SRCS}
    INCLUDE_DIRS "${PS_CORE_DIR}/include"
)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wno-unused-parameter)
