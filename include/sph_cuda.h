#pragma once

#ifdef CUDA_AVAILABLE

#include <vector>
#include <cstdint>

class Body;
class Scene;

struct CudaStreams;

bool isCudaAvailable();
void cudaUpdateDensity(std::vector<Body>& bodies);
void cudaUpdateDensityBinned(std::vector<Body>& bodies);
void cudaUpdateDensityBinnedSharedMem(std::vector<Body>& bodies);
void cudaRayTrace(uint8_t* pixels, int width, int height, const Scene& scene, bool useAdaptive);

CudaStreams* cudaCreateStreams();
void cudaDestroyStreams(CudaStreams* streams);
void cudaUpdateDensityAsync(std::vector<Body>& bodies, CudaStreams* streams);
void cudaUpdateDensityBinnedAsync(std::vector<Body>& bodies, CudaStreams* streams);
void cudaUpdateDensityBinnedSharedMemAsync(std::vector<Body>& bodies, CudaStreams* streams);
void cudaRayTraceAsync(uint8_t* pixels, int width, int height, const Scene& scene, CudaStreams* streams, bool useAdaptive);
void cudaSyncSphStream(CudaStreams* streams);
void cudaSyncRenderStream(CudaStreams* streams);

#endif // CUDA_AVAILABLE

