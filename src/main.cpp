#include "camera.h"
#include "renderer.h"
#include "vec3.h"
#include <algorithm>
#include <chrono>
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
    bool useBinning = false;
    bool useOverlap = false;
    bool useAdaptive = false;
    bool useSharedMem = false;

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
        } else if (std::strncmp(argv[i], "--binning=", 10) == 0) {
            const char* value = argv[i] + 10;
            if (std::strcmp(value, "true") == 0) {
                useBinning = true;
            } else if (std::strcmp(value, "false") == 0) {
                useBinning = false;
            } else {
                std::cerr << "Error: Unknown value for --binning: '" << value << "'\n";
                std::cerr << "Valid values: true, false\n";
                return 1;
            }
        } else if (std::strncmp(argv[i], "--overlap=", 10) == 0) {
            const char* value = argv[i] + 10;
            if (std::strcmp(value, "true") == 0) {
                useOverlap = true;
            } else if (std::strcmp(value, "false") == 0) {
                useOverlap = false;
            } else {
                std::cerr << "Error: Unknown value for --overlap: '" << value << "'\n";
                std::cerr << "Valid values: true, false\n";
                return 1;
            }
        } else if (std::strncmp(argv[i], "--adaptive=", 11) == 0) {
            const char* value = argv[i] + 11;
            if (std::strcmp(value, "true") == 0) {
                useAdaptive = true;
            } else if (std::strcmp(value, "false") == 0) {
                useAdaptive = false;
            } else {
                std::cerr << "Error: Unknown value for --adaptive: '" << value << "'\n";
                std::cerr << "Valid values: true, false\n";
                return 1;
            }
        } else if (std::strncmp(argv[i], "--shared_mem=", 13) == 0) {
            const char* value = argv[i] + 13;
            if (std::strcmp(value, "true") == 0) {
                useSharedMem = true;
            } else if (std::strcmp(value, "false") == 0) {
                useSharedMem = false;
            } else {
                std::cerr << "Error: Unknown value for --shared_mem: '" << value << "'\n";
                std::cerr << "Valid values: true, false\n";
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

    if (useBinning && !sphGPU) {
        std::cerr << "Warning: Binning only available with --sph=gpu. Disabling binning.\n";
        useBinning = false;
    }
    
    if (useSharedMem && !useBinning) {
        std::cerr << "Warning: Shared memory optimization requires --binning=true. Disabling shared_mem.\n";
        useSharedMem = false;
    }
    
    if (useOverlap && !sphGPU && !raytracingGPU) {
        std::cerr << "Warning: Overlap only useful with GPU computation. Disabling overlap.\n";
        useOverlap = false;
    }

    SceneConfig* selectedScene = &Scenes::all[0];

    std::filesystem::path outputDir = std::filesystem::current_path().parent_path() / "output";

    if (!std::filesystem::exists(outputDir.parent_path() / "CMakeLists.txt")) {
        outputDir = std::filesystem::current_path() / "output";
    }

    std::filesystem::create_directories(outputDir);

    PixelBuffer buffer(800, 600);
    Scene scene({Vec3(0, 0, 200), 65, SUN_MASS, YELLOW}, 1000, REFRESH_RATE_60FPS, sphGPU, raytracingGPU, useBinning, useOverlap, useAdaptive, useSharedMem);

    selectedScene->setup(scene, numParticles);

    std::cout << "Running scene: " << selectedScene->name << "\n";
    std::cout << "Description: " << selectedScene->description << "\n";
    std::cout << "SPH: " << (sphGPU ? "GPU" : "CPU");
    if (useBinning) {
        std::cout << " (binned";
        if (useSharedMem) std::cout << "+shared_mem";
        std::cout << ")";
    }
    std::cout << "\n";
    std::cout << "Raytracing: " << (raytracingGPU ? "GPU" : "CPU") << (useAdaptive ? " (adaptive)" : "") << "\n";
    std::cout << "Stream overlap: " << (useOverlap ? "enabled" : "disabled") << "\n";
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

    std::cout << "\nRendering:\n" << std::flush;

    double totalTime = 0.0;
    for (int frame = 0; frame < numFrames; ++frame) {
        auto frameStart = std::chrono::high_resolution_clock::now();
        
        if (useOverlap && frame > 0) {
#ifdef CUDA_AVAILABLE
            scene.syncSphStream();
#endif
        }
        
        buffer.clear(0, 0, 0);
        
        scene.draw(buffer);
        
        scene.update(REFRESH_RATE_60FPS);

        /* Uncommented the following to save a frame */
        // std::ostringstream savedFramePath;
        // savedFramePath << outputDir.string() << "/" << selectedScene->name << "_frame_"
        //          << std::setfill('0') << std::setw(4) << frame << ".png";

        // if (buffer.saveAsPNG(savedFramePath.str().c_str())) {
        //     if (frame % 10 == 0 || frame == numFrames - 1) {
        //         std::cout << "Saved " << savedFramePath.str() << " (" << (frame + 1)
        //                  << "/" << numFrames << ")\n";
        //     }
        // } else {
        //     std::cerr << "Failed to save " << savedFramePath.str() << "\n";
        // }
        
        if (useOverlap) {
#ifdef CUDA_AVAILABLE
            scene.syncRenderStream();
#endif
        }

        fwrite(buffer.pixels.data(), 1, buffer.pixels.size(), ffmpeg);
        
        auto frameEnd = std::chrono::high_resolution_clock::now();
        double frameTime = std::chrono::duration<double>(frameEnd - frameStart).count();
        totalTime += frameTime;
        
        std::cout << "Frame " << std::setw(4) << (frame + 1) << "/" << numFrames 
                  << " - " << std::fixed << std::setprecision(3) << frameTime << "s" 
                  << " ("                   << std::setprecision(1) << (1.0 / frameTime) << " FPS)\n" << std::flush;
    }
    
    if (useOverlap) {
#ifdef CUDA_AVAILABLE
        scene.syncSphStream();
        scene.syncRenderStream();
#endif
    }

    pclose(ffmpeg);
    std::cout << "\nComplete! Video saved to " << videoPath << "\n";
    std::cout << "Average frame time: " << std::fixed << std::setprecision(3) 
              << (totalTime / numFrames) << "s (" 
              << std::setprecision(1) << (numFrames / totalTime) << " FPS)\n";

    return 0;
}
