cmake_minimum_required(VERSION 3.20)
project(sphyra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Check for OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP support enabled")
endif()

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

set(CUDA_SOURCES "")
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CUDA_SOURCES src/cuda/sph_kernels.cu)
    add_compile_definitions(CUDA_AVAILABLE)
    message(STATUS "CUDA support enabled")
else()
    message(STATUS "CUDA not found - building CPU-only version")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/external)

add_executable(sphyra
    src/main.cpp
    src/core/camera.cpp
    src/core/scene.cpp
    src/core/scenes.cpp
    src/render/renderer.cpp
    ${CUDA_SOURCES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(sphyra PUBLIC OpenMP::OpenMP_CXX)
endif()

# Run Linter
add_custom_target(
    lint
    COMMAND /opt/homebrew/opt/llvm/bin/clang-tidy -p
        ${CMAKE_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/src/main.cpp
        ${CMAKE_SOURCE_DIR}/src/core/camera.cpp
        ${CMAKE_SOURCE_DIR}/src/core/scene.cpp
        ${CMAKE_SOURCE_DIR}/src/core/scenes.cpp
        ${CMAKE_SOURCE_DIR}/src/render/renderer.cpp
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "clang-tidy"
)

# Run formatter
add_custom_target(
    format
    COMMAND /opt/homebrew/opt/llvm/bin/clang-format -i
        ${CMAKE_SOURCE_DIR}/src/main.cpp
        ${CMAKE_SOURCE_DIR}/src/core/scene.cpp
        ${CMAKE_SOURCE_DIR}/src/core/scenes.cpp
        ${CMAKE_SOURCE_DIR}/src/core/camera.cpp
        ${CMAKE_SOURCE_DIR}/src/render/renderer.cpp
        ${CMAKE_SOURCE_DIR}/include/camera.h
        ${CMAKE_SOURCE_DIR}/include/renderer.h
        ${CMAKE_SOURCE_DIR}/include/constants.h
        ${CMAKE_SOURCE_DIR}/include/scenes.h
        ${CMAKE_SOURCE_DIR}/include/vec3.h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "clang-format"
)
