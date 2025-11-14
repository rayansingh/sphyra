#include "camera.h"
#include "renderer.h"
#include "vec3.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "constants.h"
#include "scene.h"
#include "scenes.h"


int main(int argc, char *argv[]) {
    int numFrames = 1;
    int numParticles = 500;
    OptimizationLevel optimization = OptimizationLevel::BASELINE;
    ComputeBackend backend = ComputeBackend::CPU;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--optimization=", 15) == 0) {
            const char* level = argv[i] + 15;
            if (std::strcmp(level, "baseline") == 0) {
                optimization = OptimizationLevel::BASELINE;
                backend = ComputeBackend::CPU;
            } else if (std::strcmp(level, "gpu_density") == 0) {
                optimization = OptimizationLevel::GPU_DENSITY;
                backend = ComputeBackend::GPU;
#ifndef CUDA_AVAILABLE
                std::cerr << "Error: gpu_density optimization requires CUDA support\n";
                std::cerr << "Please rebuild with CUDA or use --optimization=baseline\n";
                return 1;
#endif
            } else {
                std::cerr << "Error: Unknown optimization level '" << level << "'\n";
                std::cerr << "Valid levels: baseline, gpu_density\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--particles") == 0 || std::strcmp(argv[i], "-p") == 0) {
            if (i + 1 < argc) {
                numParticles = std::atoi(argv[++i]);
                if (numParticles < 1) {
                    std::cerr << "Invalid particle count. Using default: 500\n";
                    numParticles = 500;
                }
            } else {
                std::cerr << "Error: --particles requires a number\n";
                return 1;
            }
        } else {
            // Assume it's the frame count
            numFrames = std::atoi(argv[i]);
            if (numFrames < 1) {
                std::cerr << "Invalid frame count. Using default: 1\n";
                numFrames = 1;
            }
        }
    }

    SceneConfig* selectedScene = &Scenes::all[0];

    std::filesystem::path outputDir = std::filesystem::current_path().parent_path() / "output";

    if (!std::filesystem::exists(outputDir.parent_path() / "CMakeLists.txt")) {
        outputDir = std::filesystem::current_path() / "output";
    }

    std::filesystem::create_directories(outputDir);

    if (std::filesystem::exists(outputDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(outputDir)) {
            if (entry.path().extension() == ".png") {
                std::filesystem::remove(entry.path());
            }
        }
    }

    PixelBuffer buffer(800, 600);
    Scene scene({Vec3(0, 0, 200), 65, SUN_MASS, YELLOW}, 1000, REFRESH_RATE_60FPS, backend, optimization);

    selectedScene->setup(scene, numParticles);

    const char* optimizationName = (optimization == OptimizationLevel::BASELINE) ? "baseline" : "gpu_density";
    
    std::cout << "Running scene: " << selectedScene->name << "\n";
    std::cout << "Description: " << selectedScene->description << "\n";
    std::cout << "Optimization: " << optimizationName << "\n";
    std::cout << "Particles: " << numParticles << "\n";
    std::cout << "Frames: " << numFrames << "\n";
    std::cout << "Output directory: " << outputDir << "\n";
    std::cout << "\nRendering: " << std::flush;

    for (int frame = 0; frame < numFrames; ++frame) {
        buffer.clear(0, 0, 0);
        scene.draw(buffer);

        scene.update(REFRESH_RATE_60FPS);

        std::ostringstream filename;
        filename << outputDir.string() << "/" << selectedScene->name << "_frame_"
                 << std::setfill('0') << std::setw(4) << frame << ".png";

        if (buffer.saveAsPNG(filename.str().c_str())) {
            std::cout << "." << std::flush;
        } else {
            std::cerr << "\nFailed to save " << filename.str() << "\n";
        }
    }

    std::cout << "\nComplete! Frames saved to output/\n";

    if (numFrames > 1) {
        std::cout << "\nCreating video with ffmpeg...\n";
        std::ostringstream ffmpegCmd;
        ffmpegCmd << "ffmpeg -y -framerate 60 -i " << outputDir.string()
                  << "/" << selectedScene->name << "_frame_%04d.png"
                  << " -c:v libx264 -pix_fmt yuv420p -crf 18 "
                  << outputDir.string() << "/" << selectedScene->name << ".mp4";

        int result = std::system(ffmpegCmd.str().c_str());
        if (result == 0) {
            std::cout << "Video created: " << outputDir.string() << "/"
                      << selectedScene->name << ".mp4\n";
        } else {
            std::cerr << "Failed to create video. Make sure ffmpeg is installed.\n";
        }
    }

    return 0;
}
