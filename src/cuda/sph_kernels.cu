#include "sph_cuda.h"
#include "scene.h"
#include "constants.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
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

__device__ int3 getCellCoords(float x, float y, float z, float cellSize) {
    return make_int3(
        static_cast<int>(floorf(x / cellSize)),
        static_cast<int>(floorf(y / cellSize)),
        static_cast<int>(floorf(z / cellSize))
    );
}

__device__ int getCellHash(int3 cell, int gridSize) {
    int x = (cell.x % gridSize + gridSize) % gridSize;
    int y = (cell.y % gridSize + gridSize) % gridSize;
    int z = (cell.z % gridSize + gridSize) % gridSize;
    return (z * gridSize * gridSize) + (y * gridSize) + x;
}

__global__ void assignCellsKernel(ParticleData* particles, int* cellIds, int numParticles, float cellSize, int gridSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    ParticleData& p = particles[idx];
    int3 cell = getCellCoords(p.x, p.y, p.z, cellSize);
    cellIds[idx] = getCellHash(cell, gridSize);
}

__global__ void computeCellBoundsKernel(int* cellIds, int* cellStart, int* cellEnd, int numParticles, int numCells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    int cellId = cellIds[idx];
    
    if (idx == 0 || cellId != cellIds[idx - 1]) {
        cellStart[cellId] = idx;
    }
    
    if (idx == numParticles - 1 || cellId != cellIds[idx + 1]) {
        cellEnd[cellId] = idx + 1;
    }
}

__global__ void computeDensityBinnedKernel(
    ParticleData* particles,
    int* particleIndices,
    int* cellStart,
    int* cellEnd,
    double* densities,
    int numParticles,
    float cellSize,
    int gridSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    int particleIdx = particleIndices[idx];
    ParticleData& particle = particles[particleIdx];
    
    const double h2 = SMOOTHING_LENGTH * SMOOTHING_LENGTH;
    double density = 0.0;
    
    int3 cellCoord = getCellCoords(particle.x, particle.y, particle.z, cellSize);
    
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int3 neighborCell = make_int3(
                    cellCoord.x + dx,
                    cellCoord.y + dy,
                    cellCoord.z + dz
                );
                
                int cellHash = getCellHash(neighborCell, gridSize);
                int start = cellStart[cellHash];
                int end = cellEnd[cellHash];
                
                for (int j = start; j < end; j++) {
                    int neighborIdx = particleIndices[j];
                    ParticleData& neighbor = particles[neighborIdx];
                    
                    float dx = neighbor.x - particle.x;
                    float dy = neighbor.y - particle.y;
                    float dz = neighbor.z - particle.z;
                    float r2 = dx * dx + dy * dy + dz * dz;
                    
                    if (r2 < h2) {
                        double t = h2 - r2;
                        density += neighbor.mass * POLY6 * t * t * t;
                    }
                }
            }
        }
    }
    
    densities[particleIdx] = density;
}

void cudaUpdateDensityBinned(std::vector<Body>& bodies) {
    if (!isCudaAvailable()) {
        std::cerr << "Error: CUDA device not available\n";
        exit(1);
    }
    
    int numParticles = bodies.size();
    const float cellSize = SMOOTHING_LENGTH;
    const int gridSize = 128;
    const int numCells = gridSize * gridSize * gridSize;
    
    std::vector<ParticleData> hostParticles(numParticles);
    std::vector<double> hostDensities(numParticles);
    std::vector<int> hostIndices(numParticles);
    
    for (int i = 0; i < numParticles; ++i) {
        hostParticles[i].x = bodies[i].center.x;
        hostParticles[i].y = bodies[i].center.y;
        hostParticles[i].z = bodies[i].center.z;
        hostParticles[i].mass = bodies[i].mass;
        hostIndices[i] = i;
    }
    
    ParticleData* d_particles;
    int* d_cellIds;
    int* d_particleIndices;
    int* d_cellStart;
    int* d_cellEnd;
    double* d_densities;
    
    CUDA_CHECK(cudaMalloc(&d_particles, numParticles * sizeof(ParticleData)));
    CUDA_CHECK(cudaMalloc(&d_cellIds, numParticles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_particleIndices, numParticles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellStart, numCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellEnd, numCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_densities, numParticles * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_particles, hostParticles.data(), 
                          numParticles * sizeof(ParticleData), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particleIndices, hostIndices.data(), 
                          numParticles * sizeof(int), 
                          cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(d_cellStart, 0xFF, numCells * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cellEnd, 0xFF, numCells * sizeof(int)));
    
    int blockSize = 256;
    int gridSizeBlocks = (numParticles + blockSize - 1) / blockSize;
    
    assignCellsKernel<<<gridSizeBlocks, blockSize>>>(d_particles, d_cellIds, numParticles, cellSize, gridSize);
    CUDA_CHECK(cudaGetLastError());
    
    thrust::device_ptr<int> cellIds_ptr(d_cellIds);
    thrust::device_ptr<int> indices_ptr(d_particleIndices);
    thrust::sort_by_key(cellIds_ptr, cellIds_ptr + numParticles, indices_ptr);
    
    computeCellBoundsKernel<<<gridSizeBlocks, blockSize>>>(d_cellIds, d_cellStart, d_cellEnd, numParticles, numCells);
    CUDA_CHECK(cudaGetLastError());
    
    computeDensityBinnedKernel<<<gridSizeBlocks, blockSize>>>(
        d_particles, d_particleIndices, d_cellStart, d_cellEnd, 
        d_densities, numParticles, cellSize, gridSize
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(hostDensities.data(), d_densities, 
                          numParticles * sizeof(double), 
                          cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < numParticles; ++i) {
        bodies[i].density = hostDensities[i];
    }
    
    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_cellIds));
    CUDA_CHECK(cudaFree(d_particleIndices));
    CUDA_CHECK(cudaFree(d_cellStart));
    CUDA_CHECK(cudaFree(d_cellEnd));
    CUDA_CHECK(cudaFree(d_densities));
}

__global__ void computeDensityBinnedSharedMemKernel(
    ParticleData* particles,
    int* particleIndices,
    int* cellStart,
    int* cellEnd,
    double* densities,
    int numParticles,
    float cellSize,
    int gridSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    extern __shared__ ParticleData sharedParticles[];
    
    int particleIdx = particleIndices[idx];
    ParticleData p = particles[particleIdx];
    
    float px = p.x;
    float py = p.y;
    float pz = p.z;
    
    int cellX = (int)((px + 1000.0f) / cellSize);
    int cellY = (int)((py + 1000.0f) / cellSize);
    int cellZ = (int)((pz + 1000.0f) / cellSize);
    
    cellX = max(0, min(gridSize - 1, cellX));
    cellY = max(0, min(gridSize - 1, cellY));
    cellZ = max(0, min(gridSize - 1, cellZ));
    
    double density = 0.0;
    
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cellX + dx;
                int ny = cellY + dy;
                int nz = cellZ + dz;
                
                if (nx < 0 || nx >= gridSize || ny < 0 || ny >= gridSize || nz < 0 || nz >= gridSize) {
                    continue;
                }
                
                int neighborCell = nz * gridSize * gridSize + ny * gridSize + nx;
                int start = cellStart[neighborCell];
                int end = cellEnd[neighborCell];
                
                if (start == -1) continue;
                
                int cellParticleCount = end - start;
                int numChunks = (cellParticleCount + blockDim.x - 1) / blockDim.x;
                
                for (int chunk = 0; chunk < numChunks; ++chunk) {
                    int loadIdx = chunk * blockDim.x + threadIdx.x;
                    if (start + loadIdx < end) {
                        int neighborIdx = particleIndices[start + loadIdx];
                        sharedParticles[threadIdx.x] = particles[neighborIdx];
                    }
                    __syncthreads();
                    
                    int chunkSize = min((int)blockDim.x, cellParticleCount - chunk * blockDim.x);
                    for (int i = 0; i < chunkSize; ++i) {
                        ParticleData neighbor = sharedParticles[i];
                        
                        float dx_val = px - neighbor.x;
                        float dy_val = py - neighbor.y;
                        float dz_val = pz - neighbor.z;
                        float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;
                        float h2 = SMOOTHING_LENGTH * SMOOTHING_LENGTH;
                        
                        if (r2 < h2) {
                            double t = h2 - r2;
                            density += neighbor.mass * POLY6 * t * t * t;
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
    
    densities[particleIdx] = density;
}

void cudaUpdateDensityBinnedSharedMem(std::vector<Body>& bodies) {
    if (!isCudaAvailable()) {
        std::cerr << "Error: CUDA device not available\n";
        exit(1);
    }
    
    int numParticles = bodies.size();
    const float cellSize = SMOOTHING_LENGTH;
    const int gridSize = 128;
    const int numCells = gridSize * gridSize * gridSize;
    
    std::vector<ParticleData> hostParticles(numParticles);
    std::vector<double> hostDensities(numParticles);
    std::vector<int> hostIndices(numParticles);
    
    for (int i = 0; i < numParticles; ++i) {
        hostParticles[i].x = bodies[i].center.x;
        hostParticles[i].y = bodies[i].center.y;
        hostParticles[i].z = bodies[i].center.z;
        hostParticles[i].mass = bodies[i].mass;
        hostIndices[i] = i;
    }
    
    ParticleData* d_particles;
    int* d_cellIds;
    int* d_particleIndices;
    int* d_cellStart;
    int* d_cellEnd;
    double* d_densities;
    
    CUDA_CHECK(cudaMalloc(&d_particles, numParticles * sizeof(ParticleData)));
    CUDA_CHECK(cudaMalloc(&d_cellIds, numParticles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_particleIndices, numParticles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellStart, numCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellEnd, numCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_densities, numParticles * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_particles, hostParticles.data(), 
                          numParticles * sizeof(ParticleData), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particleIndices, hostIndices.data(), 
                          numParticles * sizeof(int), 
                          cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(d_cellStart, 0xFF, numCells * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cellEnd, 0xFF, numCells * sizeof(int)));
    
    int blockSize = 256;
    int gridSizeBlocks = (numParticles + blockSize - 1) / blockSize;
    
    assignCellsKernel<<<gridSizeBlocks, blockSize>>>(d_particles, d_cellIds, numParticles, cellSize, gridSize);
    CUDA_CHECK(cudaGetLastError());
    
    thrust::device_ptr<int> cellIds_ptr(d_cellIds);
    thrust::device_ptr<int> indices_ptr(d_particleIndices);
    thrust::sort_by_key(cellIds_ptr, cellIds_ptr + numParticles, indices_ptr);
    
    computeCellBoundsKernel<<<gridSizeBlocks, blockSize>>>(d_cellIds, d_cellStart, d_cellEnd, numParticles, numCells);
    CUDA_CHECK(cudaGetLastError());
    
    size_t sharedMemSize = blockSize * sizeof(ParticleData);
    computeDensityBinnedSharedMemKernel<<<gridSizeBlocks, blockSize, sharedMemSize>>>(
        d_particles, d_particleIndices, d_cellStart, d_cellEnd, 
        d_densities, numParticles, cellSize, gridSize
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(hostDensities.data(), d_densities, 
                          numParticles * sizeof(double), 
                          cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < numParticles; ++i) {
        bodies[i].density = hostDensities[i];
    }
    
    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_cellIds));
    CUDA_CHECK(cudaFree(d_particleIndices));
    CUDA_CHECK(cudaFree(d_cellStart));
    CUDA_CHECK(cudaFree(d_cellEnd));
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
    int numBodies,
    bool useAdaptive,
    bool useBinning,
    int* particleIndices,
    int* cellStart,
    int* cellEnd,
    float cellSize,
    int gridSize
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
    
    float pos_x = camera.pos_x;
    float pos_y = camera.pos_y;
    float pos_z = camera.pos_z;
    
    const float eventHorizon = bodies[0].radius * EVENT_HORIZON_MULTIPLIER;
    const float tMin = 0.001f;
    const float maxDist = MAX_GEODESIC_DISTANCE;
    const float baseStepSize = GEODESIC_STEP;
    const float sceneBoundingRadius = 800.0f;
    
    const float minStepSize = baseStepSize * 0.5f;
    const float maxStepSize = baseStepSize * 3.0f;
    const float adaptiveRadius = eventHorizon * 10.0f;
    
    uint8_t r = 0, g = 0, b = 0;
    
    for (float t = 0; t < maxDist; ) {
        float toBH_x = bodies[0].cx - pos_x;
        float toBH_y = bodies[0].cy - pos_y;
        float toBH_z = bodies[0].cz - pos_z;
        float distance = vec3_length(toBH_x, toBH_y, toBH_z);
        
        if (distance < eventHorizon) {
            break;
        }
        
        if (distance > sceneBoundingRadius) {
            break;
        }
        
        float dt;
        if (useAdaptive) {
            if (distance < adaptiveRadius) {
                dt = minStepSize;
            } else {
                float distanceFactor = fminf(1.0f, (distance - adaptiveRadius) / adaptiveRadius);
                dt = minStepSize + (maxStepSize - minStepSize) * distanceFactor;
            }
        } else {
            dt = baseStepSize;
        }
        
        // Normalize direction to black hole
        float toBH_norm_x = toBH_x / distance;
        float toBH_norm_y = toBH_y / distance;
        float toBH_norm_z = toBH_z / distance;
        
        // Calculate deflection strength
        float deflectionMag = LENSING_STRENGTH / (distance * distance);
        
        // Project deflection perpendicular to ray direction
        // Remove component parallel to ray (dot product)
        float parallel = vec3_dot(toBH_norm_x, toBH_norm_y, toBH_norm_z, ray_dx, ray_dy, ray_dz);
        float perp_x = toBH_norm_x - ray_dx * parallel;
        float perp_y = toBH_norm_y - ray_dy * parallel;
        float perp_z = toBH_norm_z - ray_dz * parallel;
        
        // Apply perpendicular deflection
        ray_dx += perp_x * deflectionMag * dt;
        ray_dy += perp_y * deflectionMag * dt;
        ray_dz += perp_z * deflectionMag * dt;
        vec3_normalize(ray_dx, ray_dy, ray_dz);
        
        t += dt;
        
        if (useBinning) {
            int cellX = (int)floorf(pos_x / cellSize);
            int cellY = (int)floorf(pos_y / cellSize);
            int cellZ = (int)floorf(pos_z / cellSize);
            
            cellX = (cellX % gridSize + gridSize) % gridSize;
            cellY = (cellY % gridSize + gridSize) % gridSize;
            cellZ = (cellZ % gridSize + gridSize) % gridSize;
            
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = cellX + dx;
                        int ny = cellY + dy;
                        int nz = cellZ + dz;
                        
                        if (nx < 0 || nx >= gridSize || ny < 0 || ny >= gridSize || nz < 0 || nz >= gridSize) {
                            continue;
                        }
                        
                        int neighborCell = nz * gridSize * gridSize + ny * gridSize + nx;
                        int start = cellStart[neighborCell];
                        int end = cellEnd[neighborCell];
                        
                        if (start == -1) continue;
                        
                        for (int idx = start; idx < end; ++idx) {
                            int k = particleIndices[idx];
                            if (k == 0) continue;
                            
                            float hit_t, hit_nx, hit_ny, hit_nz;
                            if (intersectSphere(
                                pos_x, pos_y, pos_z,
                                ray_dx, ray_dy, ray_dz,
                                bodies[k].cx, bodies[k].cy, bodies[k].cz,
                                bodies[k].radius,
                                tMin, dt * 2.0f,
                                hit_t, hit_nx, hit_ny, hit_nz
                            )) {
                                float hit_x = pos_x + ray_dx * hit_t;
                                float hit_y = pos_y + ray_dy * hit_t;
                                float hit_z = pos_z + ray_dz * hit_t;
                                
                                float light_dx = camera.light_x - hit_x;
                                float light_dy = camera.light_y - hit_y;
                                float light_dz = camera.light_z - hit_z;
                                vec3_normalize(light_dx, light_dy, light_dz);
                                
                                float diffuse = fmaxf(0.0f, vec3_dot(hit_nx, hit_ny, hit_nz, light_dx, light_dy, light_dz));
                                float ambient = 0.2f;
                                float brightness = diffuse * 0.8f + ambient;
                                
                                r = (uint8_t)fminf(255.0f, bodies[k].hue_r * brightness);
                                g = (uint8_t)fminf(255.0f, bodies[k].hue_g * brightness);
                                b = (uint8_t)fminf(255.0f, bodies[k].hue_b * brightness);
                                
                                goto done;
                            }
                        }
                    }
                }
            }
        } else {
            for (int k = 0; k < numBodies; ++k) {
                float hit_t, nx, ny, nz;
                if (intersectSphere(
                    pos_x, pos_y, pos_z,
                    ray_dx, ray_dy, ray_dz,
                    bodies[k].cx, bodies[k].cy, bodies[k].cz,
                    bodies[k].radius,
                    tMin, dt * 2.0f,
                    hit_t, nx, ny, nz
                )) {
                    float hit_x = pos_x + ray_dx * hit_t;
                    float hit_y = pos_y + ray_dy * hit_t;
                    float hit_z = pos_z + ray_dz * hit_t;
                    
                    float light_dx = camera.light_x - hit_x;
                    float light_dy = camera.light_y - hit_y;
                    float light_dz = camera.light_z - hit_z;
                    vec3_normalize(light_dx, light_dy, light_dz);
                    
                    float diffuse = fmaxf(0.0f, vec3_dot(nx, ny, nz, light_dx, light_dy, light_dz));
                    float ambient = 0.2f;
                    float brightness = diffuse * 0.8f + ambient;
                    
                    r = (uint8_t)fminf(255.0f, bodies[k].hue_r * brightness);
                    g = (uint8_t)fminf(255.0f, bodies[k].hue_g * brightness);
                    b = (uint8_t)fminf(255.0f, bodies[k].hue_b * brightness);
                    
                    goto done;
                }
            }
        }
        
        pos_x += ray_dx * dt;
        pos_y += ray_dy * dt;
        pos_z += ray_dz * dt;
    }
    
done:
    int index = (j * width + i) * 4;
    pixels[index] = r;
    pixels[index + 1] = g;
    pixels[index + 2] = b;
    pixels[index + 3] = 255;
}

void cudaRayTrace(uint8_t* pixels, int width, int height, const Scene& scene, bool useAdaptive) {
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
    int* d_particleIndices = nullptr;
    int* d_cellStart = nullptr;
    int* d_cellEnd = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_bodies, numBodies * sizeof(GPUBody)));
    CUDA_CHECK(cudaMalloc(&d_pixels, width * height * 4 * sizeof(uint8_t)));
    
    CUDA_CHECK(cudaMemcpy(d_bodies, hostBodies.data(), 
                          numBodies * sizeof(GPUBody), 
                          cudaMemcpyHostToDevice));
    
    float cellSize;
    int gridSizeDim;
    
    if (scene.useBinning) {
        float minX = 1e10f, minY = 1e10f, minZ = 1e10f;
        float maxX = -1e10f, maxY = -1e10f, maxZ = -1e10f;
        
        for (size_t i = 1; i < hostBodies.size(); ++i) {
            minX = fminf(minX, hostBodies[i].cx);
            minY = fminf(minY, hostBodies[i].cy);
            minZ = fminf(minZ, hostBodies[i].cz);
            maxX = fmaxf(maxX, hostBodies[i].cx);
            maxY = fmaxf(maxY, hostBodies[i].cy);
            maxZ = fmaxf(maxZ, hostBodies[i].cz);
        }
        
        float rangeX = maxX - minX;
        float rangeY = maxY - minY;
        float rangeZ = maxZ - minZ;
        float maxRange = fmaxf(rangeX, fmaxf(rangeY, rangeZ));
        
        float avgParticleRadius = 5.0f;
        float targetCellSize = fmaxf(avgParticleRadius * 3.0f, SMOOTHING_LENGTH);
        
        int optimalGridSize = (int)ceilf(maxRange / targetCellSize);
        optimalGridSize = fmaxf(16, fminf(128, optimalGridSize));
        
        cellSize = (maxRange + 100.0f) / optimalGridSize;
        gridSizeDim = optimalGridSize;
        const int numCells = gridSizeDim * gridSizeDim * gridSizeDim;
        std::vector<ParticleData> hostParticles(numBodies);
        std::vector<int> hostIndices(numBodies);
        
        for (int i = 0; i < numBodies; ++i) {
            hostParticles[i].x = hostBodies[i].cx;
            hostParticles[i].y = hostBodies[i].cy;
            hostParticles[i].z = hostBodies[i].cz;
            hostParticles[i].mass = hostBodies[i].radius;
            hostIndices[i] = i;
        }
        
        ParticleData* d_particles;
        int* d_cellIds;
        
        CUDA_CHECK(cudaMalloc(&d_particles, numBodies * sizeof(ParticleData)));
        CUDA_CHECK(cudaMalloc(&d_cellIds, numBodies * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_particleIndices, numBodies * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cellStart, numCells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cellEnd, numCells * sizeof(int)));
        
        CUDA_CHECK(cudaMemcpy(d_particles, hostParticles.data(), 
                              numBodies * sizeof(ParticleData), 
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_particleIndices, hostIndices.data(), 
                              numBodies * sizeof(int), 
                              cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemset(d_cellStart, 0xFF, numCells * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_cellEnd, 0xFF, numCells * sizeof(int)));
        
        int blockSize = 256;
        int gridSizeBlocks = (numBodies + blockSize - 1) / blockSize;
        
        assignCellsKernel<<<gridSizeBlocks, blockSize>>>(d_particles, d_cellIds, numBodies, cellSize, gridSizeDim);
        CUDA_CHECK(cudaGetLastError());
        
        thrust::device_ptr<int> cellIds_ptr(d_cellIds);
        thrust::device_ptr<int> indices_ptr(d_particleIndices);
        thrust::sort_by_key(cellIds_ptr, cellIds_ptr + numBodies, indices_ptr);
        
        computeCellBoundsKernel<<<gridSizeBlocks, blockSize>>>(d_cellIds, d_cellStart, d_cellEnd, numBodies, numCells);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaFree(d_particles));
        CUDA_CHECK(cudaFree(d_cellIds));
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    rayTraceKernel<<<gridSize, blockSize>>>(d_pixels, width, height, hostCamera, d_bodies, numBodies, useAdaptive,
                                             scene.useBinning, d_particleIndices, d_cellStart, d_cellEnd, cellSize, gridSizeDim);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(pixels, d_pixels, 
                          width * height * 4 * sizeof(uint8_t), 
                          cudaMemcpyDeviceToHost));
    
    if (scene.useBinning) {
        CUDA_CHECK(cudaFree(d_particleIndices));
        CUDA_CHECK(cudaFree(d_cellStart));
        CUDA_CHECK(cudaFree(d_cellEnd));
    }
    
    CUDA_CHECK(cudaFree(d_bodies));
    CUDA_CHECK(cudaFree(d_pixels));
}

struct CudaStreams {
    cudaStream_t sphStream;
    cudaStream_t renderStream;
};

CudaStreams* cudaCreateStreams() {
    if (!isCudaAvailable()) {
        return nullptr;
    }
    
    CudaStreams* streams = new CudaStreams();
    CUDA_CHECK(cudaStreamCreate(&streams->sphStream));
    CUDA_CHECK(cudaStreamCreate(&streams->renderStream));
    return streams;
}

void cudaDestroyStreams(CudaStreams* streams) {
    if (!streams) return;
    
    CUDA_CHECK(cudaStreamDestroy(streams->sphStream));
    CUDA_CHECK(cudaStreamDestroy(streams->renderStream));
    delete streams;
}

void cudaUpdateDensityAsync(std::vector<Body>& bodies, CudaStreams* streams) {
    if (!streams) {
        cudaUpdateDensity(bodies);
        return;
    }
    cudaStream_t stream = streams->sphStream;
    if (!isCudaAvailable()) {
        std::cerr << "Error: CUDA device not available\n";
        exit(1);
    }
    
    int numParticles = bodies.size();
    
    std::vector<ParticleData> hostParticles(numParticles);
    std::vector<double> hostDensities(numParticles);
    
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
    
    CUDA_CHECK(cudaMemcpyAsync(d_particles, hostParticles.data(), 
                          numParticles * sizeof(ParticleData), 
                          cudaMemcpyHostToDevice, stream));
    
    int blockSize = 256;
    int gridSize = ceil((1.0*numParticles)/blockSize);
    
    computeDensityKernel<<<gridSize, blockSize, 0, stream>>>(d_particles, d_densities, numParticles);
    
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpyAsync(hostDensities.data(), d_densities, 
                          numParticles * sizeof(double), 
                          cudaMemcpyDeviceToHost, stream));
    
    for (int i = 0; i < numParticles; ++i) {
        bodies[i].density = hostDensities[i];
    }
    
    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_densities));
}

void cudaUpdateDensityBinnedAsync(std::vector<Body>& bodies, CudaStreams* streams) {
    if (!streams) {
        cudaUpdateDensityBinned(bodies);
        return;
    }
    cudaStream_t stream = streams->sphStream;
    if (!isCudaAvailable()) {
        std::cerr << "Error: CUDA device not available\n";
        exit(1);
    }
    
    int numParticles = bodies.size();
    const float cellSize = SMOOTHING_LENGTH;
    const int gridSize = 128;
    const int numCells = gridSize * gridSize * gridSize;
    
    std::vector<ParticleData> hostParticles(numParticles);
    std::vector<double> hostDensities(numParticles);
    std::vector<int> hostIndices(numParticles);
    
    for (int i = 0; i < numParticles; ++i) {
        hostParticles[i].x = bodies[i].center.x;
        hostParticles[i].y = bodies[i].center.y;
        hostParticles[i].z = bodies[i].center.z;
        hostParticles[i].mass = bodies[i].mass;
        hostIndices[i] = i;
    }
    
    ParticleData* d_particles;
    int* d_cellIds;
    int* d_particleIndices;
    int* d_cellStart;
    int* d_cellEnd;
    double* d_densities;
    
    CUDA_CHECK(cudaMalloc(&d_particles, numParticles * sizeof(ParticleData)));
    CUDA_CHECK(cudaMalloc(&d_cellIds, numParticles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_particleIndices, numParticles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellStart, numCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellEnd, numCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_densities, numParticles * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpyAsync(d_particles, hostParticles.data(), 
                          numParticles * sizeof(ParticleData), 
                          cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_particleIndices, hostIndices.data(), 
                          numParticles * sizeof(int), 
                          cudaMemcpyHostToDevice, stream));
    
    CUDA_CHECK(cudaMemsetAsync(d_cellStart, 0xFF, numCells * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_cellEnd, 0xFF, numCells * sizeof(int), stream));
    
    int blockSize = 256;
    int gridSizeBlocks = (numParticles + blockSize - 1) / blockSize;
    
    assignCellsKernel<<<gridSizeBlocks, blockSize, 0, stream>>>(d_particles, d_cellIds, numParticles, cellSize, gridSize);
    CUDA_CHECK(cudaGetLastError());
    
    thrust::device_ptr<int> cellIds_ptr(d_cellIds);
    thrust::device_ptr<int> indices_ptr(d_particleIndices);
    thrust::sort_by_key(thrust::cuda::par.on(stream), cellIds_ptr, cellIds_ptr + numParticles, indices_ptr);
    
    computeCellBoundsKernel<<<gridSizeBlocks, blockSize, 0, stream>>>(d_cellIds, d_cellStart, d_cellEnd, numParticles, numCells);
    CUDA_CHECK(cudaGetLastError());
    
    computeDensityBinnedKernel<<<gridSizeBlocks, blockSize, 0, stream>>>(
        d_particles, d_particleIndices, d_cellStart, d_cellEnd, 
        d_densities, numParticles, cellSize, gridSize
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpyAsync(hostDensities.data(), d_densities, 
                          numParticles * sizeof(double), 
                          cudaMemcpyDeviceToHost, stream));
    
    for (int i = 0; i < numParticles; ++i) {
        bodies[i].density = hostDensities[i];
    }
    
    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_cellIds));
    CUDA_CHECK(cudaFree(d_particleIndices));
    CUDA_CHECK(cudaFree(d_cellStart));
    CUDA_CHECK(cudaFree(d_cellEnd));
    CUDA_CHECK(cudaFree(d_densities));
}

void cudaUpdateDensityBinnedSharedMemAsync(std::vector<Body>& bodies, CudaStreams* streams) {
    if (!streams) {
        cudaUpdateDensityBinnedSharedMem(bodies);
        return;
    }
    cudaStream_t stream = streams->sphStream;
    if (!isCudaAvailable()) {
        std::cerr << "Error: CUDA device not available\n";
        exit(1);
    }
    
    int numParticles = bodies.size();
    const float cellSize = SMOOTHING_LENGTH;
    const int gridSize = 128;
    const int numCells = gridSize * gridSize * gridSize;
    
    std::vector<ParticleData> hostParticles(numParticles);
    std::vector<double> hostDensities(numParticles);
    std::vector<int> hostIndices(numParticles);
    
    for (int i = 0; i < numParticles; ++i) {
        hostParticles[i].x = bodies[i].center.x;
        hostParticles[i].y = bodies[i].center.y;
        hostParticles[i].z = bodies[i].center.z;
        hostParticles[i].mass = bodies[i].mass;
        hostIndices[i] = i;
    }
    
    ParticleData* d_particles;
    int* d_cellIds;
    int* d_particleIndices;
    int* d_cellStart;
    int* d_cellEnd;
    double* d_densities;
    
    CUDA_CHECK(cudaMalloc(&d_particles, numParticles * sizeof(ParticleData)));
    CUDA_CHECK(cudaMalloc(&d_cellIds, numParticles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_particleIndices, numParticles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellStart, numCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellEnd, numCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_densities, numParticles * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpyAsync(d_particles, hostParticles.data(), 
                          numParticles * sizeof(ParticleData), 
                          cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_particleIndices, hostIndices.data(), 
                          numParticles * sizeof(int), 
                          cudaMemcpyHostToDevice, stream));
    
    CUDA_CHECK(cudaMemsetAsync(d_cellStart, 0xFF, numCells * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_cellEnd, 0xFF, numCells * sizeof(int), stream));
    
    int blockSize = 256;
    int gridSizeBlocks = (numParticles + blockSize - 1) / blockSize;
    
    assignCellsKernel<<<gridSizeBlocks, blockSize, 0, stream>>>(d_particles, d_cellIds, numParticles, cellSize, gridSize);
    CUDA_CHECK(cudaGetLastError());
    
    thrust::device_ptr<int> cellIds_ptr(d_cellIds);
    thrust::device_ptr<int> indices_ptr(d_particleIndices);
    thrust::sort_by_key(thrust::cuda::par.on(stream), cellIds_ptr, cellIds_ptr + numParticles, indices_ptr);
    
    computeCellBoundsKernel<<<gridSizeBlocks, blockSize, 0, stream>>>(d_cellIds, d_cellStart, d_cellEnd, numParticles, numCells);
    CUDA_CHECK(cudaGetLastError());
    
    size_t sharedMemSize = blockSize * sizeof(ParticleData);
    computeDensityBinnedSharedMemKernel<<<gridSizeBlocks, blockSize, sharedMemSize, stream>>>(
        d_particles, d_particleIndices, d_cellStart, d_cellEnd, 
        d_densities, numParticles, cellSize, gridSize
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpyAsync(hostDensities.data(), d_densities, 
                          numParticles * sizeof(double), 
                          cudaMemcpyDeviceToHost, stream));
    
    for (int i = 0; i < numParticles; ++i) {
        bodies[i].density = hostDensities[i];
    }
    
    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_cellIds));
    CUDA_CHECK(cudaFree(d_particleIndices));
    CUDA_CHECK(cudaFree(d_cellStart));
    CUDA_CHECK(cudaFree(d_cellEnd));
    CUDA_CHECK(cudaFree(d_densities));
}

void cudaRayTraceAsync(uint8_t* pixels, int width, int height, const Scene& scene, CudaStreams* streams, bool useAdaptive) {
    if (!streams) {
        cudaRayTrace(pixels, width, height, scene, useAdaptive);
        return;
    }
    cudaStream_t stream = streams->renderStream;
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
    int* d_particleIndices = nullptr;
    int* d_cellStart = nullptr;
    int* d_cellEnd = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_bodies, numBodies * sizeof(GPUBody)));
    CUDA_CHECK(cudaMalloc(&d_pixels, width * height * 4 * sizeof(uint8_t)));
    
    CUDA_CHECK(cudaMemcpyAsync(d_bodies, hostBodies.data(), 
                          numBodies * sizeof(GPUBody), 
                          cudaMemcpyHostToDevice, stream));
    
    const float cellSize = SMOOTHING_LENGTH;
    const int gridSizeDim = 128;
    
    if (scene.useBinning) {
        const int numCells = gridSizeDim * gridSizeDim * gridSizeDim;
        std::vector<ParticleData> hostParticles(numBodies);
        std::vector<int> hostIndices(numBodies);
        
        for (int i = 0; i < numBodies; ++i) {
            hostParticles[i].x = hostBodies[i].cx;
            hostParticles[i].y = hostBodies[i].cy;
            hostParticles[i].z = hostBodies[i].cz;
            hostParticles[i].mass = hostBodies[i].radius;
            hostIndices[i] = i;
        }
        
        ParticleData* d_particles;
        int* d_cellIds;
        
        CUDA_CHECK(cudaMalloc(&d_particles, numBodies * sizeof(ParticleData)));
        CUDA_CHECK(cudaMalloc(&d_cellIds, numBodies * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_particleIndices, numBodies * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cellStart, numCells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cellEnd, numCells * sizeof(int)));
        
        CUDA_CHECK(cudaMemcpyAsync(d_particles, hostParticles.data(), 
                              numBodies * sizeof(ParticleData), 
                              cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_particleIndices, hostIndices.data(), 
                              numBodies * sizeof(int), 
                              cudaMemcpyHostToDevice, stream));
        
        CUDA_CHECK(cudaMemsetAsync(d_cellStart, 0xFF, numCells * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(d_cellEnd, 0xFF, numCells * sizeof(int), stream));
        
        int blockSize = 256;
        int gridSizeBlocks = (numBodies + blockSize - 1) / blockSize;
        
        assignCellsKernel<<<gridSizeBlocks, blockSize, 0, stream>>>(d_particles, d_cellIds, numBodies, cellSize, gridSizeDim);
        CUDA_CHECK(cudaGetLastError());
        
        thrust::device_ptr<int> cellIds_ptr(d_cellIds);
        thrust::device_ptr<int> indices_ptr(d_particleIndices);
        thrust::sort_by_key(thrust::cuda::par.on(stream), cellIds_ptr, cellIds_ptr + numBodies, indices_ptr);
        
        computeCellBoundsKernel<<<gridSizeBlocks, blockSize, 0, stream>>>(d_cellIds, d_cellStart, d_cellEnd, numBodies, numCells);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaFree(d_particles));
        CUDA_CHECK(cudaFree(d_cellIds));
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    rayTraceKernel<<<gridSize, blockSize, 0, stream>>>(d_pixels, width, height, hostCamera, d_bodies, numBodies, useAdaptive,
                                                        scene.useBinning, d_particleIndices, d_cellStart, d_cellEnd, cellSize, gridSizeDim);
    
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpyAsync(pixels, d_pixels, 
                          width * height * 4 * sizeof(uint8_t), 
                          cudaMemcpyDeviceToHost, stream));
    
    if (scene.useBinning) {
        CUDA_CHECK(cudaFree(d_particleIndices));
        CUDA_CHECK(cudaFree(d_cellStart));
        CUDA_CHECK(cudaFree(d_cellEnd));
    }
    
    CUDA_CHECK(cudaFree(d_bodies));
    CUDA_CHECK(cudaFree(d_pixels));
}

void cudaSyncSphStream(CudaStreams* streams) {
    if (!streams) return;
    CUDA_CHECK(cudaStreamSynchronize(streams->sphStream));
}

void cudaSyncRenderStream(CudaStreams* streams) {
    if (!streams) return;
    CUDA_CHECK(cudaStreamSynchronize(streams->renderStream));
}

