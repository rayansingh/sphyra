#pragma once
#include "vec3.h"
#include <cstdint>
#include <vector>

// Pixel buffer for software rendering
class PixelBuffer {
  public:
    int width, height;
    std::vector<uint8_t> pixels; // RGBA format

    PixelBuffer(int w, int h);
    void clear(uint8_t r, uint8_t g, uint8_t b);
    void setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b);
    bool saveAsPNG(const char* filename);
};

void drawSphere3D(PixelBuffer &buffer, Vec3 center, float radius, Vec3 lightPos, Vec3 camPos, Vec3 color);
void drawLightSource(PixelBuffer &buffer, Vec3 lightPos, Vec3 camPos);
