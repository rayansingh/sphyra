#include "renderer.h"
#include <algorithm>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image_write.h"

constexpr int SCREEN_CENTER_X = 400;
constexpr int SCREEN_CENTER_Y = 300;
constexpr float LIGHT_SPHERE_RADIUS = 10.0f;

PixelBuffer::PixelBuffer(int w, int h) : width(w), height(h) {
    pixels.resize(w * h * 4);
}

void PixelBuffer::clear(uint8_t r, uint8_t g, uint8_t b) {
    for (size_t i = 0; i < pixels.size(); i += 4) {
        pixels[i] = r;
        pixels[i + 1] = g;
        pixels[i + 2] = b;
        pixels[i + 3] = 255;
    }
}

void PixelBuffer::setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || x >= width || y < 0 || y >= height)
        return;

    int index = (y * width + x) * 4;
    pixels[index] = r;
    pixels[index + 1] = g;
    pixels[index + 2] = b;
    pixels[index + 3] = 255;
}

bool PixelBuffer::saveAsPNG(const char* filename) {
    return stbi_write_png(filename, width, height, 4, pixels.data(), width * 4) != 0;
}

void drawSphere3D(PixelBuffer &buffer, Vec3 center, float radius, Vec3 lightPos, Vec3 camPos, Vec3 color) {
    int screenRadius = (int)radius;

    int cx = SCREEN_CENTER_X + (int)(center.x - camPos.x);
    int cy = SCREEN_CENTER_Y + (int)(center.y - camPos.y);

    for (int y = -screenRadius; y <= screenRadius; y++) {
        for (int x = -screenRadius; x <= screenRadius; x++) {
            float distSq = x * x + y * y;

            if (distSq <= screenRadius * screenRadius) {
                float dz = std::sqrt(std::max(0.0f, screenRadius * screenRadius - distSq));

                Vec3 surfacePoint(center.x + x * radius / screenRadius, center.y + y * radius / screenRadius,
                                  center.z + dz * radius / screenRadius);

                Vec3 normal = (surfacePoint - center).normalized();

                Vec3 lightDir = (lightPos - surfacePoint).normalized();

                float lightAmount = std::max(0.0f, normal.dot(lightDir));
                float ambient = 0.2f;
                float intensity = ambient + lightAmount * 0.8f;

                Vec3 drawColor = color * intensity;
                buffer.setPixel(cx + x, cy + y,
                               static_cast<uint8_t>(drawColor.x),
                               static_cast<uint8_t>(drawColor.y),
                               static_cast<uint8_t>(drawColor.z));
            }
        }
    }
}

void drawLightSource(PixelBuffer &buffer, Vec3 lightPos, Vec3 camPos) {
    int screenRadius = (int)LIGHT_SPHERE_RADIUS;

    int cx = SCREEN_CENTER_X + (int)(lightPos.x - camPos.x);
    int cy = SCREEN_CENTER_Y + (int)(lightPos.y - camPos.y);

    for (int y = -screenRadius; y <= screenRadius; y++) {
        for (int x = -screenRadius; x <= screenRadius; x++) {

            if (x * x + y * y <= screenRadius * screenRadius) {
                buffer.setPixel(cx + x, cy + y, 255, 255, 150);
            }
        }
    }
}
