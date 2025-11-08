#pragma once
#include "camera.h"
#include "renderer.h"
#include "vec3.h"
#include <vector>

struct BackgroundStar {
    float x, y;
    float size;
};

class Body {
  public:
    Vec3 center;
    const float radius, mass;
    Vec3 vel, acc;
    Vec3 hue;
    double density = 0.0;
    double pressure = 0.0;
    Vec3 pressureForce;
    Vec3 viscosityForce;

    Body(Vec3 center, float radius = 10, float mass = 10, Vec3 hue = {255, 255, 255});
    Body(Vec3 center, Vec3 vel, Vec3 acc, float radius = 10, float mass = 10, Vec3 hue = {255, 255, 255});

    void update(void);
    void updateAcceleration(const Body &other);
    void draw(PixelBuffer &buffer, Vec3 lightPos, Vec3 camPos, const Camera &camera, Vec3 pivot);
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

    Scene(Body sol, unsigned int numStars = 0, float updateInterval = 0.016f);

    void updateDensity();
    void updatePressure();
    void updatePressureForce();
    void updateViscosityForce();
    void update(float deltaTime);
    void draw(PixelBuffer &buffer);
};