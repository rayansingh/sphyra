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
    vel += acc;
    center += vel;
}

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
}

bool Body::intersect(const Ray& ray, float tMin, float tMax, HitRecord& rec) const {
    Vec3 oc = ray.origin - center;
    float a = ray.direction.dot(ray.direction);
    float halfB = oc.dot(ray.direction);
    float c = oc.dot(oc) - radius * radius;
    
    float discriminant = halfB * halfB - a * c;
    if (discriminant < 0) return false;
    
    float sqrtd = sqrt(discriminant);
    
    float root = (-halfB - sqrtd) / a;
    if (root < tMin || root > tMax) {
        root = (-halfB + sqrtd) / a;
        if (root < tMin || root > tMax)
            return false;
    }
    
    rec.t = root;
    rec.point = ray.at(rec.t);
    Vec3 outwardNormal = (rec.point - center) * (1.0f / radius);
    rec.setFaceNormal(ray, outwardNormal);
    rec.color = hue;
    
    return true;
}


Scene::Scene(Body sol, unsigned int numStars, float updateInterval, bool sphGPU, bool raytracingGPU, bool useBinning, bool useOverlap, bool useAdaptive, bool useSharedMem)
    : sol(sol), pivot(0, 0, 200), updateInterval(updateInterval), accumulator(0.0f), sphGPU(sphGPU), raytracingGPU(raytracingGPU), useBinning(useBinning), useOverlap(useOverlap), useAdaptive(useAdaptive), useSharedMem(useSharedMem), cudaStreams(nullptr) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distX(0.0f, 800.0f);
    std::uniform_real_distribution<float> distY(0.0f, 600.0f);
    std::uniform_real_distribution<float> distSize(1.0f, 3.0f);

    stars.reserve(numStars);
    for (unsigned int i = 0; i < numStars; i++) {
        stars.push_back({distX(gen), distY(gen), distSize(gen)});
    }
    
#ifdef CUDA_AVAILABLE
    if (useOverlap && (sphGPU || raytracingGPU)) {
        cudaStreams = cudaCreateStreams();
    }
#endif
}

Scene::~Scene() {
#ifdef CUDA_AVAILABLE
    if (cudaStreams) {
        cudaDestroyStreams(static_cast<CudaStreams*>(cudaStreams));
    }
#endif
}

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

void Scene::updateDensity() {
    if (sphGPU) {
#ifdef CUDA_AVAILABLE
        if (useOverlap && cudaStreams) {
            if (useBinning) {
                if (useSharedMem) {
                    cudaUpdateDensityBinnedSharedMemAsync(bodies, static_cast<CudaStreams*>(cudaStreams));
                } else {
                    cudaUpdateDensityBinnedAsync(bodies, static_cast<CudaStreams*>(cudaStreams));
                }
            } else {
                cudaUpdateDensityAsync(bodies, static_cast<CudaStreams*>(cudaStreams));
            }
        } else {
            if (useBinning) {
                if (useSharedMem) {
                    cudaUpdateDensityBinnedSharedMem(bodies);
                } else {
                    cudaUpdateDensityBinned(bodies);
                }
            } else {
                cudaUpdateDensity(bodies);
            }
        }
#else
        std::cerr << "Error: GPU SPH requested but CUDA support not compiled in\n";
        exit(1);
#endif
    } else {
        updateDensityCPU();
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
    if (raytracingGPU) {
#ifdef CUDA_AVAILABLE
        if (useOverlap && cudaStreams) {
            cudaRayTraceAsync(buffer.pixels.data(), buffer.width, buffer.height, *this, static_cast<CudaStreams*>(cudaStreams), useAdaptive);
        } else {
            cudaRayTrace(buffer.pixels.data(), buffer.width, buffer.height, *this, useAdaptive);
        }
#else
        std::cerr << "Error: GPU raytracing requested but CUDA support not compiled in\n";
        exit(1);
#endif
    } else {
        renderWithRayTracing(buffer, *this);
    }
}

void Scene::syncSphStream() {
#ifdef CUDA_AVAILABLE
    if (useOverlap && cudaStreams) {
        cudaSyncSphStream(static_cast<CudaStreams*>(cudaStreams));
    }
#endif
}

void Scene::syncRenderStream() {
#ifdef CUDA_AVAILABLE
    if (useOverlap && cudaStreams) {
        cudaSyncRenderStream(static_cast<CudaStreams*>(cudaStreams));
    }
#endif
}