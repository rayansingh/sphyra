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

#include "scenes.h"

constexpr float REFRESH_RATE_30FPS = 0.033f;
constexpr float REFRESH_RATE_60FPS = 0.016f;
constexpr float REFRESH_RATE_120FPS = 0.008f;

constexpr double NEWTON_GRAVITATIONAL_CONSTANT = 6.67430e-11;
constexpr double DISTANCE_SCALE = 1e9;
constexpr double TIME_SCALE = 86400.0;
constexpr double G_SCALED =
    NEWTON_GRAVITATIONAL_CONSTANT * TIME_SCALE * TIME_SCALE / (DISTANCE_SCALE * DISTANCE_SCALE * DISTANCE_SCALE);

constexpr double EARTH_MASS = 5.972e24;
constexpr double SUN_MASS = 1.989e30;

constexpr double SMOOTHING_LENGTH = 20.0;
constexpr double REST_DENSITY = 1.0;
constexpr double GAS_CONSTANT = 500.0;
constexpr double MU = 0.1;
constexpr double POLY6 = 315.0 / (64.0 * M_PI * 1.953125e15);
constexpr double SPIKYGRAD = -45.0 / (M_PI * 1.5625e10);
constexpr double VISCLAPLACIAN = 45.0 / (M_PI * 1.5625e10);

static const Vec3 BLUE(0.0f, 0.0f, 255.0f);
static const Vec3 YELLOW(255.0f,255.0f, 0.0f);
static const Vec3 RED(255.0f, 0.0f, 0.0f);
static const Vec3 WHITE(255.0f, 255.0f, 255.0f);

struct BackgroundStar {
    float x, y;
    float size;
};

class Body { // particle
  public:
    Vec3 center;
    const float radius, mass;
    Vec3 vel, acc;
    Vec3 hue;
    double density = 0.0;
    double pressure = 0.0;
    Vec3 pressureForce;
    Vec3 viscosityForce;

    Body(Vec3 center, float radius = 10, float mass = 10, Vec3 hue = {255, 255, 255})
        : center(center), radius(radius), mass(mass), hue(hue) {
        vel = Vec3(0, 0, 0);
        acc = Vec3(0, 0, 0);
    };
    Body(Vec3 center, Vec3 vel, Vec3 acc, float radius = 10, float mass = 10, Vec3 hue = {255, 255, 255})
        : center(center), vel(vel), acc(acc), radius(radius), mass(mass), hue(hue) {};

    void update(void) {
        center += vel;
        vel += acc;
    }

    // Find and add acceleration
    // DUE TO gravity towards other body
    // AND pressure force and viscosity force
    void updateAcceleration(const Body &other) {
        Vec3 rVec = other.center - center;
        float r = rVec.length();
        if (r < 1e-6f) {
            acc = Vec3();
            return;
        }
        acc = Vec3();

        float GM = 2000.0f;
        acc += rVec * (GM / (r * r * r));
    }

    void draw(PixelBuffer &buffer, Vec3 lightPos, Vec3 camPos, const Camera &camera, Vec3 pivot) {
        Vec3 rotatedCenter = camera.rotatePoint(center - pivot) + pivot;
        Vec3 rotatedLight = camera.rotatePoint(lightPos - pivot) + pivot;
        drawSphere3D(buffer, rotatedCenter, radius, rotatedLight, camPos, hue);
    }
};

class Scene {
  public:
    Camera camera;
    Body sol;
    std::vector<Body> bodies;
    std::vector<BackgroundStar> stars;
    Vec3 pivot;           // Rotation pivot point
    float updateInterval; // Time between physics updates (seconds)
    float accumulator;    // Accumulated time for physics

    Scene(Body sol, unsigned int numStars = 0, float updateInterval = 0.016f)
        : sol(sol), pivot(0, 0, 200), updateInterval(updateInterval), accumulator(0.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distX(0.0f, 800.0f);
        std::uniform_real_distribution<float> distY(0.0f, 600.0f);
        std::uniform_real_distribution<float> distSize(1.0f, 3.0f);

        stars.reserve(numStars);
        for (unsigned int i = 0; i < numStars; i++) {
            stars.push_back({distX(gen), distY(gen), distSize(gen)});
        }
    }

    // Update Density for all particles, O(n2)
    void updateDensity() {
        const float h2 = SMOOTHING_LENGTH * SMOOTHING_LENGTH;
        for (auto &body : bodies) {
            body.density = 0.0;
            for (auto &neighbor : bodies) {
                Vec3 rVec = neighbor.center - body.center;
                float r2 = rVec.lengthSquared();
                if (r2 < h2) {
                    float t = h2 - r2;
                    body.density += neighbor.mass * POLY6 * t * t * t;
                }
            }
        }
    }

    void updatePressure() {
        for (auto &body : bodies) {
            body.pressure = GAS_CONSTANT * (body.density - REST_DENSITY);
        }
    }

    void updatePressureForce() {
        for (auto &body : bodies) {
            body.pressureForce = Vec3();
            for (auto& neighbor : bodies) {
                if (&body == &neighbor) continue;
                Vec3 rVec = neighbor.center - body.center;
                float r = rVec.length();
                if (r < SMOOTHING_LENGTH && r > 1e-6f) {
                    float t = SMOOTHING_LENGTH - r;
                    Vec3 grad = rVec.normalized() * SPIKYGRAD * t * t;
                    body.pressureForce += grad * (neighbor.mass * (-0.5) * (body.pressure + neighbor.pressure) / neighbor.density);
                }
            }
        }
    }

    void updateViscosityForce() {
        for (auto &body : bodies) {
            body.viscosityForce = Vec3();
            for (auto &neighbor : bodies) {
                if (&body == &neighbor) continue;
                Vec3 rVec = neighbor.center - body.center;
                float t = SMOOTHING_LENGTH - rVec.length();
                if (t > 0) {
                    body.viscosityForce += (neighbor.vel - body.vel) * (neighbor.mass / neighbor.density * VISCLAPLACIAN * t);
                }
            }
        }
    }

    void update(float deltaTime) {
        accumulator += deltaTime;

        while (accumulator >= updateInterval) {
            updateDensity();
            updatePressure();
            updatePressureForce();
            updateViscosityForce();
            for (auto &body : bodies) {
                body.updateAcceleration(sol);
                body.update();
            }
            accumulator -= updateInterval;
        }
    }

    void draw(PixelBuffer &buffer) {
        std::vector<Body*> sortedBodies;
        sortedBodies.reserve(bodies.size() + 1);
        sortedBodies.push_back(&sol);
        for (auto &body : bodies) {
            sortedBodies.push_back(&body);
        }

        std::cout << "Sorting " << sortedBodies.size() << " particles..." << std::flush;
        std::sort(sortedBodies.begin(), sortedBodies.end(), [&](const Body* a, const Body* b) {
            Vec3 rotatedA = camera.rotatePoint(a->center - pivot) + pivot;
            Vec3 rotatedB = camera.rotatePoint(b->center - pivot) + pivot;
            return rotatedA.z > rotatedB.z;
        });
        std::cout << " done\n";

        std::cout << "Rendering particles: [" << std::flush;
        size_t total = sortedBodies.size();
        size_t progressStep = total / 50;
        if (progressStep == 0) progressStep = 1;

        for (size_t i = 0; i < sortedBodies.size(); i++) {
            sortedBodies[i]->draw(buffer, camera.lightPos, camera.position, camera, pivot);
            if (i % progressStep == 0) {
                std::cout << "#" << std::flush;
            }
        }
        std::cout << "] done\n" << std::flush;
    }
};

static const Vec3 ORANGE(255.0f, 165.0f, 0.0f);

void Scenes::accretionDisk(Scene& scene) {
    scene.bodies.clear();

    scene.sol.center = Vec3(400, 300, 200);
    scene.sol.hue = Vec3(0, 0, 0);
    scene.pivot = Vec3(400, 300, 200);

    scene.camera.angleX = -M_PI / 2.5f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distR(100.0f, 220.0f);
    std::uniform_real_distribution<float> distTheta(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> distZ(-8.0f, 8.0f);

    for (int i = 0; i < 5000/*25000*/; i++) {
        float r = distR(gen);
        float theta = distTheta(gen);
        float z = distZ(gen);

        Vec3 pos(400 + r * std::cos(theta), 300 + r * std::sin(theta), 200 + z);

        float GM = 2000.0f;
        float orbitalSpeed = std::sqrt(GM / r);

        Vec3 vel(-orbitalSpeed * std::sin(theta), orbitalSpeed * std::cos(theta), 0);

        float t = (r - 100.0f) / 120.0f;
        Vec3 color;
        if (t < 0.5f) {
            color = ORANGE * (1.0f - t/0.5f) + YELLOW * (t/0.5f);
        } else {
            color = YELLOW * (1.0f - (t-0.5f)/0.5f) + RED * ((t-0.5f)/0.5f);
        }

        scene.bodies.push_back({pos, vel, Vec3(0, 0, 0), 0.5, 5.0f, color});
    }
}

std::vector<SceneConfig> Scenes::all = {
    {"disk", "Accretion disk around central star", Scenes::accretionDisk}
};

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
