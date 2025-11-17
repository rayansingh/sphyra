#pragma once

#ifdef CUDA_AVAILABLE

#include <vector>
#include <cstdint>

class Body;
class Scene;

bool isCudaAvailable();
void cudaUpdateDensity(std::vector<Body>& bodies);
void cudaRayTrace(uint8_t* pixels, int width, int height, const Scene& scene);

#endif // CUDA_AVAILABLE

