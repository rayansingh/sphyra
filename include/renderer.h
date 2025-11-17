#pragma once
#include "vec3.h"
#include <cstdint>
#include <vector>

class Scene;

class PixelBuffer {
  public:
    int width, height;
    std::vector<uint8_t> pixels;

    PixelBuffer(int w, int h);
    void clear(uint8_t r, uint8_t g, uint8_t b);
    void setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b);
    bool saveAsPNG(const char* filename);
};

Vec3 rayTrace(const Ray& ray, const Scene& scene);
void renderWithRayTracing(PixelBuffer& buffer, const Scene& scene);
void renderWithRayTracingGPU(PixelBuffer& buffer, const Scene& scene);
