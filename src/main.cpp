#include "camera.h"
#include "renderer.h"
#include "vec3.h"
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

constexpr float REFRESH_RATE_30FPS = 0.033f;
constexpr float REFRESH_RATE_60FPS = 0.016f;
constexpr float REFRESH_RATE_120FPS = 0.008f;

constexpr double NEWTON_GRAVITATIONAL_CONSTANT = 6.67430e-11; // m³/(kg·s²)
constexpr double DISTANCE_SCALE = 1e9;                        // 1 pixel = 1 million km = 1e9 meters
constexpr double TIME_SCALE = 86400.0;                        // 1 sim second = 1 day
constexpr double G_SCALED =
    NEWTON_GRAVITATIONAL_CONSTANT * TIME_SCALE * TIME_SCALE / (DISTANCE_SCALE * DISTANCE_SCALE * DISTANCE_SCALE);

constexpr double EARTH_MASS = 5.972e24;
constexpr double SUN_MASS = 1.989e30;


// 'h' in SPH, neighbor search radius, max distance at which one particle can influence another
// TODO: might need to be tuned SMOOTHING_LENGTH, REST_DENSITY, GAS_CONSTANT, MU
constexpr double SMOOTHING_LENGTH = 2.5;
constexpr double REST_DENSITY = 1.0;   // reference density
constexpr double GAS_CONSTANT = 2000.0;   // controls stiffness of pressure
constexpr double MU = 0.1; // viscosity of fluid
constexpr double POLY6 = 315.0 / (64.0 * M_PI * 1.953125e15); // density, std::pow(SMOOTHING_LENGTH,9)=1.953125e15
constexpr double SPIKYGRAD = -45.0 / (M_PI * 1.5625e10); // pressure force, std::pow(SMOOTHING_LENGTH,6)=1.5625e10
constexpr double VISCLAPLACIAN = 45.0 / (M_PI * 1.5625e10);

static const Vec3 BLUE(0.0f, 0.0f, 255.0f);
static const Vec3 YELLOW(255.0f,255.0f, 0.0f);

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
        float r2 = rVec.lengthSquared();
        double eps = 1e-3; // softening to avoid blow-ups
        double inv_r3 = 1.0 / std::pow(r2 + eps*eps, 1.5);
        acc = Vec3(); // Reset

        // float aMag = G_SCALED * other.mass / (r2 * std::sqrt(r2)); // G*M/(r^3); Physics
        // numerically safe
        acc += rVec * G_SCALED * other.mass * inv_r3; // acc from external force
        float inv_mass = 1.0f / mass;
        acc += pressureForce * inv_mass;
        acc += viscosityForce * inv_mass;
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
        for (const auto &star : stars) {
            int size = static_cast<int>(star.size);
            for (int dy = 0; dy < size; dy++) {
                for (int dx = 0; dx < size; dx++) {
                    buffer.setPixel(static_cast<int>(star.x) + dx, static_cast<int>(star.y) + dy, 255, 255, 255);
                }
            }
        }

        sol.draw(buffer, camera.lightPos, camera.position, camera, pivot);

        for (auto &body : bodies) {
            body.draw(buffer, camera.lightPos, camera.position, camera, pivot);
        }

        Vec3 rotatedLight = camera.rotatePoint(camera.lightPos - pivot) + pivot;
        drawLightSource(buffer, rotatedLight, camera.position);
    }
};

int main(int argc, char *argv[]) {
    int numFrames = 1;
    if (argc > 1) {
        numFrames = std::atoi(argv[1]);
        if (numFrames < 1) {
            std::cerr << "Invalid frame count. Using default: 1\n";
            numFrames = 1;
        }
    }

    std::filesystem::create_directories("output");

    PixelBuffer buffer(800, 600);
    Scene scene({Vec3(0, 0, 200), 65, SUN_MASS, YELLOW}, 1000, REFRESH_RATE_60FPS);

    float earthDist = 150.0f;
    float earthVel = 30000.0f / DISTANCE_SCALE * TIME_SCALE;
    scene.bodies.push_back({Vec3(earthDist, 0, 200), Vec3(0, earthVel, 0),
                            Vec3(0, 0, 0),
                            6,
                            EARTH_MASS, BLUE});

    for (int frame = 0; frame < numFrames; ++frame) {
        scene.update(REFRESH_RATE_60FPS);

        buffer.clear(0, 0, 0);
        scene.draw(buffer);

        std::ostringstream filename;
        filename << "output/frame_" << std::setfill('0') << std::setw(4) << frame << ".png";

        if (buffer.saveAsPNG(filename.str().c_str())) {
            std::cout << "Saved " << filename.str() << "\n";
        } else {
            std::cerr << "Failed to save " << filename.str() << "\n";
        }
    }

    return 0;
}
