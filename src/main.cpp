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

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [frames]\n\n";
    std::cout << "Generates frames of an accretion disk simulation.\n";
    std::cout << "If frames > 1, automatically creates an MP4 video using ffmpeg.\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << progName << " 1           # Single frame\n";
    std::cout << "  " << progName << " 100         # 100 frames + video\n";
    std::cout << "  " << progName << "             # Default (1 frame)\n";
}

int main(int argc, char *argv[]) {
    int numFrames = 1;

    if (argc > 1) {
        if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        }
        numFrames = std::atoi(argv[1]);
        if (numFrames < 1) {
            std::cerr << "Invalid frame count. Using default: 1\n";
            numFrames = 1;
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
    Scene scene({Vec3(0, 0, 200), 65, SUN_MASS, YELLOW}, 1000, REFRESH_RATE_60FPS);

    selectedScene->setup(scene);

    std::cout << "Running scene: " << selectedScene->name << "\n";
    std::cout << "Description: " << selectedScene->description << "\n";
    std::cout << "Frames: " << numFrames << "\n";
    std::cout << "Output directory: " << outputDir << "\n";

    for (int frame = 0; frame < numFrames; ++frame) {
        buffer.clear(0, 0, 0);
        scene.draw(buffer);

        scene.update(REFRESH_RATE_60FPS);

        std::ostringstream filename;
        filename << outputDir.string() << "/" << selectedScene->name << "_frame_"
                 << std::setfill('0') << std::setw(4) << frame << ".png";

        if (buffer.saveAsPNG(filename.str().c_str())) {
            if (frame % 10 == 0 || frame == numFrames - 1) {
                std::cout << "Saved " << filename.str() << " (" << (frame + 1)
                         << "/" << numFrames << ")\n";
            }
        } else {
            std::cerr << "Failed to save " << filename.str() << "\n";
        }
    }

    std::cout << "Complete! Frames saved to output/\n";

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
