cmake_minimum_required(VERSION 3.10)
project(SinArr)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -Wall")

add_executable(sin_float task.c)
add_executable(sin_double task.c)

target_compile_definitions(sin_double PRIVATE USE_DOUBLE)