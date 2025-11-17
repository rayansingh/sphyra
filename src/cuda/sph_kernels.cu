#include "sph_cuda.h"
#include "scene.h"
#include "constants.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// GPU Particle Data
struct ParticleData {
    float x, y, z;
    float mass;
};

// Density computation
__global__ void computeDensityKernel(ParticleData* particles, double* densities, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numParticles) return;
    
    const double h2 = SMOOTHING_LENGTH * SMOOTHING_LENGTH;
    double density = 0.0;
    
    ParticleData& particle = particles[idx];
    
    // Loop over all neighbors (brute force for now)
    for (int j = 0; j < numParticles; ++j) {
        ParticleData& neighbor = particles[j];
        
        float dx = neighbor.x - particle.x;
        float dy = neighbor.y - particle.y;
        float dz = neighbor.z - particle.z;
        float r2 = dx * dx + dy * dy + dz * dz;
        
        if (r2 < h2) {
            double t = h2 - r2;
            density += neighbor.mass * POLY6 * t * t * t;
        }
    }
    
    densities[idx] = density;
}

// Check CUDA availability
bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        return false;
    }
    return true;
}

// Wrapper function for density computation
void cudaUpdateDensity(std::vector<Body>& bodies) {
    if (!isCudaAvailable()) {
        std::cerr << "Error: CUDA device not available\n";
        exit(1);
    }
    
    int numParticles = bodies.size();
    
    std::vector<ParticleData> hostParticles(numParticles);
    std::vector<double> hostDensities(numParticles);
    
    // Copy particle data to host arrays
    for (int i = 0; i < numParticles; ++i) {
        hostParticles[i].x = bodies[i].center.x;
        hostParticles[i].y = bodies[i].center.y;
        hostParticles[i].z = bodies[i].center.z;
        hostParticles[i].mass = bodies[i].mass;
    }
    
    ParticleData* d_particles;
    double* d_densities;
    
    CUDA_CHECK(cudaMalloc(&d_particles, numParticles * sizeof(ParticleData)));
    CUDA_CHECK(cudaMalloc(&d_densities, numParticles * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_particles, hostParticles.data(), 
                          numParticles * sizeof(ParticleData), 
                          cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = ceil((1.0*numParticles)/blockSize);
    
    computeDensityKernel<<<gridSize, blockSize>>>(d_particles, d_densities, numParticles);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(hostDensities.data(), d_densities, 
                          numParticles * sizeof(double), 
                          cudaMemcpyDeviceToHost));
    
    // Update densities
    for (int i = 0; i < numParticles; ++i) {
        bodies[i].density = hostDensities[i];
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_densities));
}

struct GPUBody {
    float cx, cy, cz;
    float radius;
    float hue_r, hue_g, hue_b;
};

struct GPUCamera {
    float pos_x, pos_y, pos_z;
    float light_x, light_y, light_z;
    float llc_x, llc_y, llc_z;
    float hor_x, hor_y, hor_z;
    float ver_x, ver_y, ver_z;
};

__device__ float vec3_dot(float ax, float ay, float az, float bx, float by, float bz) {
    return ax * bx + ay * by + az * bz;
}

__device__ float vec3_length(float x, float y, float z) {
    return sqrtf(x * x + y * y + z * z);
}

__device__ void vec3_normalize(float& x, float& y, float& z) {
    float len = vec3_length(x, y, z);
    if (len > 0.0f) {
        x /= len;
        y /= len;
        z /= len;
    }
}

__device__ bool intersectSphere(
    float ray_ox, float ray_oy, float ray_oz,
    float ray_dx, float ray_dy, float ray_dz,
    float sphere_x, float sphere_y, float sphere_z,
    float radius,
    float tMin, float tMax,
    float& t_out, float& nx_out, float& ny_out, float& nz_out
) {
    float oc_x = ray_ox - sphere_x;
    float oc_y = ray_oy - sphere_y;
    float oc_z = ray_oz - sphere_z;
    
    float a = vec3_dot(ray_dx, ray_dy, ray_dz, ray_dx, ray_dy, ray_dz);
    float halfB = vec3_dot(oc_x, oc_y, oc_z, ray_dx, ray_dy, ray_dz);
    float c = vec3_dot(oc_x, oc_y, oc_z, oc_x, oc_y, oc_z) - radius * radius;
    
    float discriminant = halfB * halfB - a * c;
    if (discriminant < 0) return false;
    
    float sqrtd = sqrtf(discriminant);
    float root = (-halfB - sqrtd) / a;
    
    if (root < tMin || root > tMax) {
        root = (-halfB + sqrtd) / a;
        if (root < tMin || root > tMax)
            return false;
    }
    
    t_out = root;
    
    float hit_x = ray_ox + ray_dx * root;
    float hit_y = ray_oy + ray_dy * root;
    float hit_z = ray_oz + ray_dz * root;
    
    nx_out = (hit_x - sphere_x) / radius;
    ny_out = (hit_y - sphere_y) / radius;
    nz_out = (hit_z - sphere_z) / radius;
    
    return true;
}

__global__ void rayTraceKernel(
    uint8_t* pixels,
    int width, int height,
    GPUCamera camera,
    GPUBody* bodies,
    int numBodies
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= width || j >= height) return;
    
    float u = float(i) / (width - 1);
    float v = 1.0f - float(j) / (height - 1);
    
    float pixel_x = camera.llc_x + camera.hor_x * u + camera.ver_x * v;
    float pixel_y = camera.llc_y + camera.hor_y * u + camera.ver_y * v;
    float pixel_z = camera.llc_z + camera.hor_z * u + camera.ver_z * v;
    
    float ray_dx = pixel_x - camera.pos_x;
    float ray_dy = pixel_y - camera.pos_y;
    float ray_dz = pixel_z - camera.pos_z;
    vec3_normalize(ray_dx, ray_dy, ray_dz);
    
    const float tMin = 0.001f;
    const float tMax = 1e10f;
    
    float closest_t = tMax;
    float hit_nx = 0, hit_ny = 0, hit_nz = 0;
    float hit_r = 0, hit_g = 0, hit_b = 0;
    bool hitAnything = false;
    
    for (int k = 0; k < numBodies; ++k) {
        float t, nx, ny, nz;
        if (intersectSphere(
            camera.pos_x, camera.pos_y, camera.pos_z,
            ray_dx, ray_dy, ray_dz,
            bodies[k].cx, bodies[k].cy, bodies[k].cz,
            bodies[k].radius,
            tMin, closest_t,
            t, nx, ny, nz
        )) {
            closest_t = t;
            hit_nx = nx;
            hit_ny = ny;
            hit_nz = nz;
            hit_r = bodies[k].hue_r;
            hit_g = bodies[k].hue_g;
            hit_b = bodies[k].hue_b;
            hitAnything = true;
        }
    }
    
    uint8_t r, g, b;
    
    if (hitAnything) {
        float hit_x = camera.pos_x + ray_dx * closest_t;
        float hit_y = camera.pos_y + ray_dy * closest_t;
        float hit_z = camera.pos_z + ray_dz * closest_t;
        
        float light_dx = camera.light_x - hit_x;
        float light_dy = camera.light_y - hit_y;
        float light_dz = camera.light_z - hit_z;
        vec3_normalize(light_dx, light_dy, light_dz);
        
        float diffuse = fmaxf(0.0f, vec3_dot(hit_nx, hit_ny, hit_nz, light_dx, light_dy, light_dz));
        
        float ambient = 0.2f;
        float brightness = diffuse * 0.8f + ambient;
        
        r = (uint8_t)fminf(255.0f, hit_r * brightness);
        g = (uint8_t)fminf(255.0f, hit_g * brightness);
        b = (uint8_t)fminf(255.0f, hit_b * brightness);
    } else {
        r = g = b = 0;
    }
    
    int index = (j * width + i) * 4;
    pixels[index] = r;
    pixels[index + 1] = g;
    pixels[index + 2] = b;
    pixels[index + 3] = 255;
}

void cudaRayTrace(uint8_t* pixels, int width, int height, const Scene& scene) {
    if (!isCudaAvailable()) {
        std::cerr << "Error: CUDA device not available\n";
        exit(1);
    }
    
    int numBodies = scene.bodies.size() + 1;
    std::vector<GPUBody> hostBodies(numBodies);
    
    hostBodies[0].cx = scene.sol.center.x;
    hostBodies[0].cy = scene.sol.center.y;
    hostBodies[0].cz = scene.sol.center.z;
    hostBodies[0].radius = scene.sol.radius;
    hostBodies[0].hue_r = scene.sol.hue.x;
    hostBodies[0].hue_g = scene.sol.hue.y;
    hostBodies[0].hue_b = scene.sol.hue.z;
    
    for (size_t i = 0; i < scene.bodies.size(); ++i) {
        hostBodies[i + 1].cx = scene.bodies[i].center.x;
        hostBodies[i + 1].cy = scene.bodies[i].center.y;
        hostBodies[i + 1].cz = scene.bodies[i].center.z;
        hostBodies[i + 1].radius = scene.bodies[i].radius;
        hostBodies[i + 1].hue_r = scene.bodies[i].hue.x;
        hostBodies[i + 1].hue_g = scene.bodies[i].hue.y;
        hostBodies[i + 1].hue_b = scene.bodies[i].hue.z;
    }
    
    GPUCamera hostCamera;
    hostCamera.pos_x = scene.camera.position.x;
    hostCamera.pos_y = scene.camera.position.y;
    hostCamera.pos_z = scene.camera.position.z;
    hostCamera.light_x = scene.camera.lightPos.x;
    hostCamera.light_y = scene.camera.lightPos.y;
    hostCamera.light_z = scene.camera.lightPos.z;
    hostCamera.llc_x = scene.camera.lowerLeftCorner.x;
    hostCamera.llc_y = scene.camera.lowerLeftCorner.y;
    hostCamera.llc_z = scene.camera.lowerLeftCorner.z;
    hostCamera.hor_x = scene.camera.horizontal.x;
    hostCamera.hor_y = scene.camera.horizontal.y;
    hostCamera.hor_z = scene.camera.horizontal.z;
    hostCamera.ver_x = scene.camera.vertical.x;
    hostCamera.ver_y = scene.camera.vertical.y;
    hostCamera.ver_z = scene.camera.vertical.z;
    
    GPUBody* d_bodies;
    uint8_t* d_pixels;
    
    CUDA_CHECK(cudaMalloc(&d_bodies, numBodies * sizeof(GPUBody)));
    CUDA_CHECK(cudaMalloc(&d_pixels, width * height * 4 * sizeof(uint8_t)));
    
    CUDA_CHECK(cudaMemcpy(d_bodies, hostBodies.data(), 
                          numBodies * sizeof(GPUBody), 
                          cudaMemcpyHostToDevice));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    rayTraceKernel<<<gridSize, blockSize>>>(d_pixels, width, height, hostCamera, d_bodies, numBodies);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(pixels, d_pixels, 
                          width * height * 4 * sizeof(uint8_t), 
                          cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_bodies));
    CUDA_CHECK(cudaFree(d_pixels));
}

