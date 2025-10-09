#include "camera.h"
#include "renderer.h"
#include "vec3.h"
#include <SDL2/SDL.h>
#include <random>
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
constexpr float SMOOTHING_LENGTH = 50.0f;
constexpr float REST_DENSITY = 1000.0f;   // reference density, might be varied for each later
constexpr float GAS_CONSTANT = 2000.0f;   // controls stiffness of pressure, might be varied for each later
constexpr float POLY6 = 315.0f / (64.0f * M_PI * 1.953125e15); // density, std::pow(SMOOTHING_LENGTH,9)=1.953125e15
constexpr float SPIKYGRAD = -45.0f / (M_PI * 1.5625e10); // pressure force, std::pow(SMOOTHING_LENGTH,6)=1.5625e10
constexpr double INV_TEMP = 1.0e-24; // Slow down the particle for observation

static const Vec3 BLUE(0.0f, 0.0f, 255.0f);
static const Vec3 YELLOW(255.0f,255.0f, 0.0f);

struct BackgroundStar {
    float x, y;
    float size;
};

class Body {
  public:
    Vec3 center;
    float radius, mass;
    Vec3 vel, acc;
    Vec3 hue;
    float density = 0.0f;
    float pressure = 0.0f;

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

    // Find and add acceleration due to gravity towards other body
    void updateAcceleration(const Body &other) {
        Vec3 rVec = other.center - center;
        float r2 = rVec.lengthSquared();

        if (r2 < 1e-6f)
            return;

        float aMag = G_SCALED * other.mass / (r2 * std::sqrt(r2)); // G*M/(r^3)
        acc += rVec * aMag;
    }

    void draw(SDL_Renderer *renderer, Vec3 lightPos, Vec3 camPos, const Camera &camera, Vec3 pivot) {
        Vec3 rotatedCenter = camera.rotatePoint(center - pivot) + pivot;
        Vec3 rotatedLight = camera.rotatePoint(lightPos - pivot) + pivot;
        drawSphere3D(renderer, rotatedCenter, radius, rotatedLight, camPos, hue);
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

    // Compute Density and Pressure for particles, O(n2)
    void computeDensityPressure() {
        const float h2 = SMOOTHING_LENGTH * SMOOTHING_LENGTH;

        for (auto &body : bodies) {
            body.density = 0.0f;
            for (auto &neighbor : bodies) {
                Vec3 rVec = neighbor.center - body.center;
                float r2 = rVec.lengthSquared();
                if (r2 < h2) {
                    float t = h2 - r2;
                    body.density += neighbor.mass * POLY6 * t * t * t;
                }
            }
            body.pressure = GAS_CONSTANT * (body.density - REST_DENSITY);
        }
    }

    // Compute Force -> affect acc
    void computeForces() {
        for (auto &body : bodies) {
            Vec3 pressureForce(0,0,0);
            for (auto& neighbor : bodies) {
                if (&body == &neighbor) continue;
                Vec3 rVec = neighbor.center - body.center;
                float r = rVec.length();
                if (r < SMOOTHING_LENGTH && r > 1e-6f) {
                    float t = SMOOTHING_LENGTH - r;
                    Vec3 grad = rVec.normalized() * SPIKYGRAD * t * t;
                    pressureForce += grad * (body.pressure / body.density + neighbor.pressure / neighbor.density);
                }
            }
            body.acc += pressureForce * (-body.mass) * INV_TEMP;
        }
    }

    void update(float deltaTime) {
        accumulator += deltaTime;

        while (accumulator >= updateInterval) {
            //computeDensityPressure();
            for (auto &body : bodies) {
                body.acc = Vec3(0, 0, 0); // Reset
                body.updateAcceleration(sol);
            //computeForces();
            for (auto &body : bodies)
                body.update();
            }
            accumulator -= updateInterval;
        }
    }

    void draw(SDL_Renderer *renderer) {
        // Draw stars first (background)
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        for (const auto &star : stars) {
            int size = static_cast<int>(star.size);
            for (int dy = 0; dy < size; dy++) {
                for (int dx = 0; dx < size; dx++) {
                    SDL_RenderDrawPoint(renderer, static_cast<int>(star.x) + dx, static_cast<int>(star.y) + dy);
                }
            }
        }

        // Draw sun
        sol.draw(renderer, camera.lightPos, camera.position, camera, pivot);

        // Draw bodies
        for (auto &body : bodies) {
            body.draw(renderer, camera.lightPos, camera.position, camera, pivot);
        }

        // Draw light source
        Vec3 rotatedLight = camera.rotatePoint(camera.lightPos - pivot) + pivot;
        drawLightSource(renderer, rotatedLight, camera.position);
    }
};

int main(int argc, char *argv[]) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *window = SDL_CreateWindow("sphyra", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, 0);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    // Sun: mass = 1.989e30 kg, radius 65 pixels
    Scene scene({Vec3(0, 0, 200), 65, SUN_MASS, YELLOW}, 1000, REFRESH_RATE_60FPS);
    bool running = true;
    SDL_Event event;

    // Earth-like planet
    // Distance: 150 million km from sun = 150 pixels
    // Orbital velocity: ~30 km/s
    float earthDist = 150.0f;
    float earthVel = 30000.0f / DISTANCE_SCALE * TIME_SCALE;               // scaled velocity
    scene.bodies.push_back({Vec3(earthDist, 0, 200), Vec3(0, earthVel, 0), // perpendicular to cause orbit
                            Vec3(0, 0, 0),
                            6, // radius
                            EARTH_MASS, BLUE});

    uint32_t lastTime = SDL_GetTicks();

    while (running) {
        // Calculate delta time
        uint32_t currentTime = SDL_GetTicks();
        float deltaTime = (currentTime - lastTime) / 1000.0f;
        lastTime = currentTime;

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
            scene.camera.handleInput(event);

            // Check for right click to place bodies
            if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_RIGHT) {
                int mouseX, mouseY;
                SDL_GetMouseState(&mouseX, &mouseY);

                // Convert screen coords to world coords (in camera space)
                float worldX = (mouseX - 400) + scene.camera.position.x;
                float worldY = (mouseY - 300) + scene.camera.position.y;
                float worldZ = scene.pivot.z; // Same depth as sun

                // Inverse rotation to get world pos
                Vec3 clickPos(worldX, worldY, worldZ);
                Vec3 relativePos = clickPos - scene.pivot;

                // Apply inverse rotation
                float cosY = std::cos(-scene.camera.angleY);
                float sinY = std::sin(-scene.camera.angleY);
                Vec3 rotY(relativePos.x * cosY - relativePos.z * sinY, relativePos.y,
                          relativePos.x * sinY + relativePos.z * cosY);

                float cosX = std::cos(-scene.camera.angleX);
                float sinX = std::sin(-scene.camera.angleX);
                Vec3 actualPos(rotY.x, rotY.y * cosX - rotY.z * sinX, rotY.y * sinX + rotY.z * cosX);

                // Create and add to bodies list
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<float> velDist(-5.0f, 5.0f);
                std::uniform_real_distribution<float> accDist(0.0f, 0.0f);
                std::uniform_real_distribution<float> massDist(10.0f, 1.0e20f);

                scene.bodies.push_back({actualPos + scene.pivot, Vec3(velDist(gen), velDist(gen), velDist(gen)),
                                        Vec3(accDist(gen), accDist(gen), accDist(gen)), 10, massDist(gen)});
            }
        }

        // Update physics
        scene.update(deltaTime);

        // Render
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        scene.draw(renderer);

        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
