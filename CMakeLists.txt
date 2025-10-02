cmake_minimum_required(VERSION 3.20)
project(sphyra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# Main executable (Week 1: Basic particles)
add_executable(sphyra
    src/main.cpp
    src/core/camera.cpp
    src/render/renderer.cpp
)

target_link_libraries(sphyra ${SDL2_LIBRARIES})
