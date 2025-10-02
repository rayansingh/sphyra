#include <SDL2/SDL.h>
#include "camera.h"
#include "renderer.h"
#include "vec3.h"

int main(int argc, char* argv[]) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("sphyra", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    Camera camera;
    bool running = true;
    SDL_Event event;

    Vec3 sphereCenter(0, 0, 200);

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
            camera.handleInput(event);
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        drawSphere3D(renderer, sphereCenter, 100, camera.lightPos, camera.position);
        drawLightSource(renderer, camera.lightPos, camera.position);

        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
