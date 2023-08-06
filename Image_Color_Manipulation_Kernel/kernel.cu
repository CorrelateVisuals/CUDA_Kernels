#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>

#include "STB_Image_Load.h"
#include "STB_Image_Write.h"

struct alignas(16) Pixel {
    unsigned char r, g, b, a;
};
const int pixelSizeBytes = sizeof(Pixel);
const int channels = pixelSizeBytes / sizeof(int);
const unsigned char maxIntensity = 255;

const struct alignas(16) greyScaleRGB {
    const float r = 0.2126f;
    const float g = 0.7152f;
    const float b = 0.0722f;
};

const int blockDimension = 32;

__global__ void threadsOnImage(unsigned char* imageRGBA) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t id = y * blockDim.x * gridDim.x + x;

    Pixel* pPixel = (Pixel*)&imageRGBA[id * channels];
    greyScaleRGB weight;
    unsigned char pixelValue = (unsigned char)(pPixel->r * weight.r + pPixel->g * weight.g + pPixel->b * weight.b);
    unsigned char offset = 50;
    int a_value = pixelValue + offset;
    pPixel->a = static_cast<unsigned char>(a_value < 0 ? 0 : (a_value > maxIntensity ? maxIntensity : a_value));

    int intensityX = threadIdx.x % blockDim.x + blockIdx.x;
    int intensityY = threadIdx.y % blockDim.y + blockIdx.y; 
    int intensity = intensityY + intensityX;
    pPixel->r = static_cast<unsigned char>( intensity );
    pPixel->g = static_cast<unsigned char>(x);
    pPixel->b = static_cast<unsigned char>(y);
}

__global__ void imageToGreyscale(unsigned char* imageRGBA) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t id = y * blockDim.x * gridDim.x + x;

    Pixel* pPixel = (Pixel*)&imageRGBA[id * channels];
    greyScaleRGB weight;
    unsigned char pixelValue = (unsigned char)(pPixel->r * weight.r + pPixel->g * weight.g + pPixel->b * weight.b);
    pPixel->r = pixelValue;
    pPixel->g = pixelValue;
    pPixel->b = pixelValue;
    pPixel->a = maxIntensity;
}

void copyDataAndWriteToDisk(unsigned char* pImageDataGPU, unsigned char* imageData, int width, int height, int channels, const std::string& fileNameOut, int strideBytes, const std::string& suffix) {
    std::cout << "Copy data from GPU ... ";
    assert(cudaMemcpy(imageData, pImageDataGPU, width * height * channels, cudaMemcpyDeviceToHost) == cudaSuccess);
    std::cout << "DONE" << std::endl;

    std::cout << "Writing png to disk ... ";
    std::string baseFileName = fileNameOut.substr(0, fileNameOut.find_last_of("."));
    std::string newFileName = baseFileName + suffix;
    stbi_write_png(newFileName.c_str(), width, height, channels, imageData, strideBytes);
    std::cout << "Done " << fileNameOut << " saved" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cout << "Usage: Image Color Manipulation <filename>" << std::endl; return -1; }

    int width, height, componentCount;
    std::cout << "Loading png file ... ";
    unsigned char* imageData = stbi_load(argv[1], &width, &height, &componentCount, channels);
    if (!imageData) { std::cout << "Failed to open \"" << argv[1] << "\"";  return -1; }
    std::cout << "Done" << std::endl;

    std::cout << "Copy data to GPU ... ";
    if (width % 32 || height % 32) { std::cout << "Width or height is not devidible by 32; leaked memory of imageData"; return -1; }
    unsigned char* pImageDataGPU = nullptr;
    assert(cudaMalloc(&pImageDataGPU, width * height * channels) == cudaSuccess);
    assert(cudaMemcpy(pImageDataGPU, imageData, width * height * channels, cudaMemcpyHostToDevice) == cudaSuccess);

    dim3 blockSize(blockDimension, blockDimension, 1);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    std::string fileNameOut = argv[1];
    int strideBytes = channels * width;

    std::cout << "Running CUDA kernel ... ";
    imageToGreyscale<<<gridSize, blockSize>>>(pImageDataGPU);
    std::cout << "DONE" << std::endl;
    copyDataAndWriteToDisk(pImageDataGPU, imageData, width, height, channels, fileNameOut, strideBytes, "_grey.jpg");

    std::cout << "Running CUDA kernel ... ";
    threadsOnImage<<<gridSize, blockSize>>>(pImageDataGPU);
    std::cout << "DONE" << std::endl;
    copyDataAndWriteToDisk(pImageDataGPU, imageData, width, height, channels, fileNameOut, strideBytes, "_threads.jpg");

    cudaFree(pImageDataGPU);
    stbi_image_free(imageData);

    return 0;
}
