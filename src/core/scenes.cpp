#include <random>
#include "constants.h"
#include "scene.h"
#include "scenes.h"

void Scenes::accretionDisk(Scene& scene, int numParticles) {
    scene.bodies.clear();

    scene.sol.center = Vec3(400, 300, 200);
    scene.sol.hue = Vec3(0, 0, 0);
    scene.pivot = Vec3(400, 300, 200);

    scene.camera.position = Vec3(400, -450, 380);
    scene.camera.angleX = -M_PI / 2.0f - 0.55f;
    scene.camera.fov = 70.0f;
    scene.camera.updateCameraVectors();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distR(150.0f, 330.0f);
    std::uniform_real_distribution<float> distTheta(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> distZ(-8.0f, 8.0f);

    for (int i = 0; i < numParticles; i++) {
        float r = distR(gen);
        float theta = distTheta(gen);
        float z = distZ(gen);

        Vec3 pos(400 + r * std::cos(theta), 300 + r * std::sin(theta), 200 + z);

        float GM = 2000.0f;
        float orbitalSpeed = std::sqrt(GM / r);

        Vec3 vel(-orbitalSpeed * std::sin(theta), orbitalSpeed * std::cos(theta), 0);

        float t = (r - 150.0f) / 180.0f;
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