#include "camera.h"

Camera::Camera() : position(0, 0, 0), lightPos(200, -200, 300) {}

void Camera::handleInput(const SDL_Event &event) {
    if (event.type == SDL_KEYDOWN) {
        switch (event.key.keysym.sym) {
        // Camera controls
        case SDLK_LEFT:
            position.x -= 10;
            break;
        case SDLK_RIGHT:
            position.x += 10;
            break;
        case SDLK_UP:
            position.y -= 10;
            break;
        case SDLK_DOWN:
            position.y += 10;
            break;
        case SDLK_w:
            position.z += 10;
            break;
        case SDLK_s:
            position.z -= 10;
            break;
        // Light source controls
        case SDLK_i:
            lightPos.y -= 20;
            break;
        case SDLK_k:
            lightPos.y += 20;
            break;
        case SDLK_j:
            lightPos.x -= 20;
            break;
        case SDLK_l:
            lightPos.x += 20;
            break;
        }
    }
}
