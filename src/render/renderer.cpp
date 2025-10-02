#include "renderer.h"
#include <algorithm>
#include <cmath>

constexpr int SCREEN_CENTER_X = 400;         // Screen center X value (800/2)
constexpr int SCREEN_CENTER_Y = 300;         // Screen center Y value (600/2)
constexpr float LIGHT_SPHERE_RADIUS = 15.0f; // Light source radius

void drawSphere3D(SDL_Renderer *renderer, Vec3 center, float radius, Vec3 lightPos, Vec3 camPos) {
    int screenRadius = (int)radius;

    // Find sphere center on screen
    int cx = SCREEN_CENTER_X + (int)(center.x - camPos.x);
    int cy = SCREEN_CENTER_Y + (int)(center.y - camPos.y);

    // For every point facing the camera (screen), calculate lighting value
    // TODO: Do this with raytracing / vectorized / GPU instructions
    for (int y = -screenRadius; y <= screenRadius; y++) {
        for (int x = -screenRadius; x <= screenRadius; x++) {
            float distSq = x * x + y * y; // D^2 = x^2 + y^2

            if (distSq <= screenRadius * screenRadius) {
                // Find surface point corresponding to pixel (intersection)
                float dz = std::sqrt(std::max(0.0f, screenRadius * screenRadius - distSq));

                Vec3 surfacePoint(center.x + x * radius / screenRadius, center.y + y * radius / screenRadius,
                                  center.z + dz * radius / screenRadius);

                // Calculate surface normal vector
                Vec3 normal = (surfacePoint - center).normalized();

                // Calculate direction from surface point to light
                Vec3 lightDir = (lightPos - surfacePoint).normalized();

                // Calculate how much of surface faces light
                float lightAmount = std::max(0.0f, normal.dot(lightDir));
                float ambient = 0.2f;
                float intensity = ambient + lightAmount * 0.8f;

                int brightness = (int)(intensity * 255);
                SDL_SetRenderDrawColor(renderer, brightness, brightness, brightness, 255);
                SDL_RenderDrawPoint(renderer, cx + x, cy + y);
            }
        }
    }
}

void drawLightSource(SDL_Renderer *renderer, Vec3 lightPos, Vec3 camPos) {
    int screenRadius = (int)LIGHT_SPHERE_RADIUS;

    // Find light position on screen
    int cx = SCREEN_CENTER_X + (int)(lightPos.x - camPos.x);
    int cy = SCREEN_CENTER_Y + (int)(lightPos.y - camPos.y);

    for (int y = -screenRadius; y <= screenRadius; y++) {
        for (int x = -screenRadius; x <= screenRadius; x++) {

            // Draw light source circle
            if (x * x + y * y <= screenRadius * screenRadius) {
                SDL_SetRenderDrawColor(renderer, 255, 255, 150, 255);
                SDL_RenderDrawPoint(renderer, cx + x, cy + y);
            }
        }
    }
}
