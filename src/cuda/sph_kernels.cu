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

