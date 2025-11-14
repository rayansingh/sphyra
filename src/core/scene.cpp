#include <iostream>
#include <random>
#include <vector>
#include "camera.h"
#include "renderer.h"
#include "vec3.h"
#include "constants.h"
#include "scene.h"
#include <algorithm>

#ifdef CUDA_AVAILABLE
#include "sph_cuda.h"
#endif


Body::Body(Vec3 center, float radius, float mass, Vec3 hue)
    : center(center), radius(radius), mass(mass), hue(hue) {
    vel = Vec3(0, 0, 0);
    acc = Vec3(0, 0, 0);
}
Body::Body(Vec3 center, Vec3 vel, Vec3 acc, float radius, float mass, Vec3 hue)
    : center(center), vel(vel), acc(acc), radius(radius), mass(mass), hue(hue) {};

void Body::update(void) {
    // Symplectic Euler integration
    vel += acc;
    center += vel;
}

// Find and add acceleration
// DUE TO gravity towards other body
// AND pressure force and viscosity force
void Body::updateAcceleration(const Body &other) {
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

void Body::draw(PixelBuffer &buffer, Vec3 lightPos, Vec3 camPos, const Camera &camera, Vec3 pivot) {
    Vec3 rotatedCenter = camera.rotatePoint(center - pivot) + pivot;
    Vec3 rotatedLight = camera.rotatePoint(lightPos - pivot) + pivot;
    drawSphere3D(buffer, rotatedCenter, radius, rotatedLight, camPos, hue);
}


Scene::Scene(Body sol, unsigned int numStars, float updateInterval, ComputeBackend backend, OptimizationLevel optimization)
    : sol(sol), pivot(0, 0, 200), updateInterval(updateInterval), accumulator(0.0f), backend(backend), optimization(optimization) {
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

// CPU implementation of density update - O(n2)
void Scene::updateDensityCPU() {
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

// Dispatcher for density update
void Scene::updateDensity() {
    // Use optimization level to determine which implementation to use
    switch (optimization) {
        case OptimizationLevel::BASELINE:
            // Baseline: CPU O(n²) neighbor search
            updateDensityCPU();
            break;
            
        case OptimizationLevel::GPU_DENSITY:
            // GPU-accelerated density calculation
#ifdef CUDA_AVAILABLE
            cudaUpdateDensity(bodies);
#else
            std::cerr << "Error: GPU density optimization requested but CUDA support not compiled in\n";
            exit(1);
#endif
            break;
            
        default:
            updateDensityCPU();
            break;
    }
}

void Scene::updatePressure() {
    for (auto &body : bodies) {
        body.pressure = GAS_CONSTANT * (body.density - REST_DENSITY);
    }
}

void Scene::updatePressureForce() {
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

void Scene::updateViscosityForce() {
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

void Scene::update(float deltaTime) {
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

void Scene::draw(PixelBuffer &buffer) {
    std::vector<Body*> sortedBodies;
    sortedBodies.reserve(bodies.size() + 1);
    sortedBodies.push_back(&sol);
    for (auto &body : bodies) {
        sortedBodies.push_back(&body);
    }

    std::sort(sortedBodies.begin(), sortedBodies.end(), [&](const Body* a, const Body* b) {
        Vec3 rotatedA = camera.rotatePoint(a->center - pivot) + pivot;
        Vec3 rotatedB = camera.rotatePoint(b->center - pivot) + pivot;
        return rotatedA.z > rotatedB.z;
    });

    for (size_t i = 0; i < sortedBodies.size(); i++) {
        sortedBodies[i]->draw(buffer, camera.lightPos, camera.position, camera, pivot);
    }
}