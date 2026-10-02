# Host build: core library, unit tests and the simulator.
# The ESP32-P4 firmware is a separate ESP-IDF project in firmware/.
cmake_minimum_required(VERSION 3.16)
project(pyrosight_host C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
add_compile_options(-Wall -Wextra -Wshadow -Wno-unused-parameter)

file(GLOB PS_CORE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/core/src/*.c)
add_library(pyrosight_core STATIC ${PS_CORE_SRC})
target_include_directories(pyrosight_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/include)
target_link_libraries(pyrosight_core PUBLIC m)

enable_testing()
file(GLOB PS_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/core/tests/test_*.c)
foreach(t ${PS_TESTS})
  get_filename_component(name ${t} NAME_WE)
  add_executable(${name} ${t})
  target_link_libraries(${name} pyrosight_core)
  add_test(NAME ${name} COMMAND ${name})
endforeach()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/sim/CMakeLists.txt)
  add_subdirectory(sim)
endif()
