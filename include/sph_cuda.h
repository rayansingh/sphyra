#pragma once

#ifdef CUDA_AVAILABLE

#include <vector>

class Body;

bool isCudaAvailable();
void cudaUpdateDensity(std::vector<Body>& bodies);

#endif // CUDA_AVAILABLE

