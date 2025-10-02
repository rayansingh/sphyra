#include "camera.h"
#include "renderer.h"
#include "vec3.h"
#include <SDL2/SDL.h>
#include <vector>

class Body {
  public:
    Vec3 center;
    float radius;
    float mass;

    Vec3 vel;
    Vec3 acc;

    Body(Vec3 center, float radius = 10, float mass = 10) : center(center), radius(radius), mass(mass) {
        vel = Vec3(0, 0, 0);
        acc = Vec3(0, 0, 0);
    };
    Body(Vec3 center, Vec3 vel = Vec3(0, 0, 0), Vec3 acc = Vec3(0, 0, 0), float radius = 10, float mass = 10)
        : center(center), vel(vel), acc(acc), radius(radius), mass(mass) {};

    void draw(SDL_Renderer *renderer, Vec3 lightPos, Vec3 camPos) {
        drawSphere3D(renderer, center, radius, lightPos, camPos);
    }
};

int main(int argc, char *argv[]) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *window = SDL_CreateWindow("sphyra", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, 0);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    Camera camera;
    bool running = true;
    SDL_Event event;

    std::vector<Body> bodies;

    // Sun
    bodies.push_back({Vec3(0, 0, 200), 65});

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
            camera.handleInput(event);

            // Check for click events
            if (event.type == SDL_MOUSEBUTTONDOWN) {
                int mouseX, mouseY;
                SDL_GetMouseState(&mouseX, &mouseY);

                // Convert clicked pixel to coords
                float worldX = (mouseX - 400) + camera.position.x;
                float worldY = (mouseY - 300) + camera.position.y;

                // Create and add to bodies list
                bodies.push_back({Vec3(worldX, worldY, 200), 10});
            }
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        for (auto &body : bodies) {
            body.draw(renderer, camera.lightPos, camera.position);
        }
        drawLightSource(renderer, camera.lightPos, camera.position);

        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
