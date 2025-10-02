#pragma once
#include "vec3.h"
#include <SDL2/SDL.h>

struct Camera {
    Vec3 position;
    Vec3 lightPos;
    float angleX, angleY; // Rotation angles
    bool dragging;
    int lastMouseX, lastMouseY;

    Camera();
    void handleInput(const SDL_Event &event);
    Vec3 rotatePoint(const Vec3 &point) const;
};
