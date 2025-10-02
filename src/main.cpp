#include "camera.h"
#include "renderer.h"
#include "vec3.h"
#include <SDL2/SDL.h>
#include <vector>
#include <random>

struct BackgroundStar {
    float x, y;
};

class Body {
  public:
    Vec3 center;
    float radius, mass;
    Vec3 vel, acc;

    Body(Vec3 center, float radius = 10, float mass = 10) : center(center), radius(radius), mass(mass) {
        vel = Vec3(0, 0, 0);
        acc = Vec3(0, 0, 0);
    };
    Body(Vec3 center, Vec3 vel, Vec3 acc, float radius = 10, float mass = 10)
        : center(center), vel(vel), acc(acc), radius(radius), mass(mass) {};

    void draw(SDL_Renderer *renderer, Vec3 lightPos, Vec3 camPos) {
        drawSphere3D(renderer, center, radius, lightPos, camPos);
    }
};

class Scene {
  public:
    Camera camera;
    std::vector<Body> bodies;
    std::vector<BackgroundStar> stars;

    Scene(unsigned int numStars = 0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distX(0.0f, 800.0f);
        std::uniform_real_distribution<float> distY(0.0f, 600.0f);

        stars.reserve(numStars);
        for (unsigned int i = 0; i < numStars; i++) {
            stars.push_back({distX(gen), distY(gen)});
        }
    }

    void draw(SDL_Renderer *renderer) {
        // Draw stars first (background)
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        for (const auto &star : stars) {
            SDL_RenderDrawPoint(renderer, static_cast<int>(star.x), static_cast<int>(star.y));
        }

        // Draw bodies
        for (auto &body : bodies) {
            body.draw(renderer, camera.lightPos, camera.position);
        }

        // Draw light source
        drawLightSource(renderer, camera.lightPos, camera.position);
    }
};

int main(int argc, char *argv[]) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *window = SDL_CreateWindow("sphyra", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, 0);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    Scene scene(100);
    bool running = true;
    SDL_Event event;

    // Sun
    scene.bodies.push_back({Vec3(0, 0, 200), 65});

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
            scene.camera.handleInput(event);

            // Check for click events
            if (event.type == SDL_MOUSEBUTTONDOWN) {
                int mouseX, mouseY;
                SDL_GetMouseState(&mouseX, &mouseY);

                // Convert clicked pixel to coords
                float worldX = (mouseX - 400) + scene.camera.position.x;
                float worldY = (mouseY - 300) + scene.camera.position.y;

                // Create and add to bodies list
                scene.bodies.push_back({Vec3(worldX, worldY, 200), 10});
            }
        }

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
