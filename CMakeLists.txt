cmake_minimum_required(VERSION 3.20)
project(sphyra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/external)

add_executable(sphyra
    src/main.cpp
    src/core/camera.cpp
    src/render/renderer.cpp
)

# Run Linter
add_custom_target(
    lint
    COMMAND /opt/homebrew/opt/llvm/bin/clang-tidy -p
        ${CMAKE_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/src/main.cpp
        ${CMAKE_SOURCE_DIR}/src/core/camera.cpp
        ${CMAKE_SOURCE_DIR}/src/render/renderer.cpp
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "clang-tidy"
)

# Run formatter
add_custom_target(
    format
    COMMAND /opt/homebrew/opt/llvm/bin/clang-format -i
        ${CMAKE_SOURCE_DIR}/src/main.cpp
        ${CMAKE_SOURCE_DIR}/src/core/camera.cpp
        ${CMAKE_SOURCE_DIR}/src/render/renderer.cpp
        ${CMAKE_SOURCE_DIR}/include/camera.h
        ${CMAKE_SOURCE_DIR}/include/renderer.h
        ${CMAKE_SOURCE_DIR}/include/vec3.h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "clang-format"
)
