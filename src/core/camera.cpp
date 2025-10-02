#include "camera.h"
#include <SDL_keycode.h>
#include <cmath>

constexpr float MOUSE_SENSITIVITY = 0.005f;

Camera::Camera() : position(0, 0, 0), lightPos(200, -200, 300), angleX(0), angleY(0), dragging(false) {}

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
    } else if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
        dragging = true;
        lastMouseX = event.button.x;
        lastMouseY = event.button.y;
    } else if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT) {
        dragging = false;
    } else if (event.type == SDL_MOUSEMOTION && dragging) {
        int deltaX = event.motion.x - lastMouseX;
        int deltaY = event.motion.y - lastMouseY;

        angleY += deltaX * MOUSE_SENSITIVITY;
        angleX += deltaY * MOUSE_SENSITIVITY;

        lastMouseX = event.motion.x;
        lastMouseY = event.motion.y;
    }
}

// Rotate point around origin using camera's angles
Vec3 Camera::rotatePoint(const Vec3 &point) const {
    // Rotate around Y axis
    float cosY = std::cos(angleY);
    float sinY = std::sin(angleY);
    Vec3 rotY(point.x * cosY - point.z * sinY, point.y, point.x * sinY + point.z * cosY);

    // Rotate around X axis
    float cosX = std::cos(angleX);
    float sinX = std::sin(angleX);
    return Vec3(rotY.x, rotY.y * cosX - rotY.z * sinX, rotY.y * sinX + rotY.z * cosX);
}
