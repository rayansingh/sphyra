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
    bool sphGPU = false;
    bool raytracingGPU = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--sph=", 6) == 0) {
            const char* value = argv[i] + 6;
            if (std::strcmp(value, "gpu") == 0) {
                sphGPU = true;
#ifndef CUDA_AVAILABLE
                std::cerr << "Error: --sph=gpu requires CUDA support\n";
                return 1;
#endif
            } else if (std::strcmp(value, "cpu") == 0) {
                sphGPU = false;
            } else {
                std::cerr << "Error: Unknown value for --sph: '" << value << "'\n";
                std::cerr << "Valid values: cpu, gpu\n";
                return 1;
            }
        } else if (std::strncmp(argv[i], "--raytracing=", 13) == 0) {
            const char* value = argv[i] + 13;
            if (std::strcmp(value, "gpu") == 0) {
                raytracingGPU = true;
#ifndef CUDA_AVAILABLE
                std::cerr << "Error: --raytracing=gpu requires CUDA support\n";
                return 1;
#endif
            } else if (std::strcmp(value, "cpu") == 0) {
                raytracingGPU = false;
            } else {
                std::cerr << "Error: Unknown value for --raytracing: '" << value << "'\n";
                std::cerr << "Valid values: cpu, gpu\n";
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

    PixelBuffer buffer(800, 600);
    Scene scene({Vec3(0, 0, 200), 65, SUN_MASS, YELLOW}, 1000, REFRESH_RATE_60FPS, sphGPU, raytracingGPU);

    selectedScene->setup(scene, numParticles);

    std::cout << "Running scene: " << selectedScene->name << "\n";
    std::cout << "Description: " << selectedScene->description << "\n";
    std::cout << "SPH: " << (sphGPU ? "GPU" : "CPU") << "\n";
    std::cout << "Raytracing: " << (raytracingGPU ? "GPU" : "CPU") << "\n";
    std::cout << "Particles: " << numParticles << "\n";
    std::cout << "Frames: " << numFrames << "\n";
    std::cout << "Output directory: " << outputDir << "\n";

    // Open ffmpeg pipe
    std::ostringstream ffmpegCmd;
    std::string videoPath = outputDir.string() + "/" + selectedScene->name + ".mp4";
    ffmpegCmd << "ffmpeg -y -f rawvideo -pix_fmt rgba " << "-s 800x600 -r 60 -i - " << "-c:v libx264 -pix_fmt yuv420p -crf 18 " << videoPath << " 2>/dev/null";
    
    FILE* ffmpeg = popen(ffmpegCmd.str().c_str(), "w");
    if (!ffmpeg) {
        std::cerr << "Failed to open ffmpeg pipe\n";
        return 1;
    }

    std::cout << "\nRendering: " << std::flush;

    for (int frame = 0; frame < numFrames; ++frame) {
        buffer.clear(0, 0, 0);
        scene.draw(buffer);
        scene.update(REFRESH_RATE_60FPS);

        fwrite(buffer.pixels.data(), 1, buffer.pixels.size(), ffmpeg);
        std::cout << "." << std::flush;
    }

    pclose(ffmpeg);
    std::cout << "\nComplete! Video saved to " << videoPath << "\n";

    return 0;
}
