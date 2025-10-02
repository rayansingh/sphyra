#pragma once
#include <SDL2/SDL.h>
#include "vec3.h"

struct Camera {
    Vec3 position;
    Vec3 lightPos;

    Camera();
    void handleInput(const SDL_Event& event);
};
