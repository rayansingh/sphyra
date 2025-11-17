#include "renderer.h"
#include "scene.h"
#include <algorithm>
#include <cmath>

#ifdef CUDA_AVAILABLE
#include "sph_cuda.h"
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image_write.h"

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

Vec3 rayTrace(const Ray& ray, const Scene& scene) {
    const float tMin = 0.001f;
    const float tMax = 1e10f;
    
    HitRecord closestHit;
    closestHit.t = tMax;
    bool hitAnything = false;
    
    if (scene.sol.intersect(ray, tMin, closestHit.t, closestHit)) {
        hitAnything = true;
    }
    
    for (const auto& body : scene.bodies) {
        if (body.intersect(ray, tMin, closestHit.t, closestHit)) {
            hitAnything = true;
        }
    }
    
    if (hitAnything) {
        Vec3 lightDir = (scene.camera.lightPos - closestHit.point).normalized();
        float diffuse = std::max(0.0f, closestHit.normal.dot(lightDir));
        
        float ambient = 0.2f;
        float brightness = diffuse * 0.8f + ambient;
        Vec3 color = closestHit.color * brightness;
        
        return Vec3(
            std::min(255.0f, color.x),
            std::min(255.0f, color.y),
            std::min(255.0f, color.z)
        );
    }
    
    return Vec3(0, 0, 0);
}

void renderWithRayTracing(PixelBuffer& buffer, const Scene& scene) {
    for (int j = 0; j < buffer.height; ++j) {
        for (int i = 0; i < buffer.width; ++i) {
            float u = float(i) / (buffer.width - 1);
            float v = 1.0f - float(j) / (buffer.height - 1);
            
            Ray ray = scene.camera.getRay(u, v);
            Vec3 color = rayTrace(ray, scene);
            
            buffer.setPixel(i, j, 
                static_cast<uint8_t>(color.x),
                static_cast<uint8_t>(color.y),
                static_cast<uint8_t>(color.z)
            );
        }
    }
}

void renderWithRayTracingGPU(PixelBuffer& buffer, const Scene& scene) {
#ifdef CUDA_AVAILABLE
    cudaRayTrace(buffer.pixels.data(), buffer.width, buffer.height, scene);
#else
    renderWithRayTracing(buffer, scene);
#endif
}
