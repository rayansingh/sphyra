#include <iostream>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "../external/stb_image.h"

const float EPS = 1e-4;
const float ERR_THR = 1e-5;

bool loadImage(const std::string& filename, std::vector<uint8_t>& pixels, int& width, int& height) {
    int n_channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &n_channels, 4);
    if (!data) return false;
    pixels.assign(data, data + width * height * 4);
    stbi_image_free(data);
    return true;
}

int countPixelDifferences(const std::vector<uint8_t>& cpuImg, const std::vector<uint8_t>& gpuImg) {
    assert(cpuImg.size() == gpuImg.size());
    int error = 0;
    for (int i = 0; i < cpuImg.size(); i++) {
        if (fabs(cpuImg[i] - gpuImg[i]) / (cpuImg[i] + EPS) > ERR_THR) {
            error++;
        }
    }
    return error;
}

float computeMSE(const std::vector<uint8_t>& cpuImg, const std::vector<uint8_t>& gpuImg) {
    assert(cpuImg.size() == gpuImg.size());
    double mse = 0.0;
    for (size_t i = 0; i < cpuImg.size(); i++) {
        double diff = double(cpuImg[i]) - double(gpuImg[i]);
        mse += diff * diff;
    }
    return float(mse / cpuImg.size());
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " cpu_frame.png gpu_frame.png\n";
        return 1;
    }

    std::string cpuPath = argv[1];
    std::string gpuPath = argv[2];

    std::vector<uint8_t> cpuPixels, gpuPixels;
    int w1, h1, w2, h2;

    if (!loadImage(cpuPath, cpuPixels, w1, h1)) return 1;
    if (!loadImage(gpuPath, gpuPixels, w2, h2)) return 1;

    if (w1 != w2 || h1 != h2) {
        std::cerr << "Size mismatch: cpu (" << w1 << "x" << h1 << ") vs gpu (" << w2 << "x" << h2 << ")\n";
        return 1;
    }

    int pixeldiff = countPixelDifferences(cpuPixels, gpuPixels);
    float mse  = computeMSE(cpuPixels, gpuPixels);

    std::cout << "pixeldiff = " << pixeldiff << "\n";
    std::cout << "MSE  = " << mse  << "\n";

    if (pixeldiff > 0) {
        std::cout << "Test failed with " << pixeldiff << " differing pixels.\n";
    } else {
        std::cout << "Frames are identical.\n";
    }

    return 0;
}