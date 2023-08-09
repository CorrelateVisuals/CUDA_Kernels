
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

const size_t device = 0;

int width = 64, height = 64;
float* pDevice;
size_t pitch;



__global__ void addKernel(float* pDevice, size_t pitch, int width, int height){
    for (int r = 0; r < height; r++) {
        float* row = (float*)((char*)pDevice + r * pitch);
        for (int c = 0; c < width; c++) {
            float* element = row[c];
        }
    }
}

int main()
{
    cudaMallocPitch(&pDevice, &pitch, width * sizeof(float), height);

    cudaError_t cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {fprintf (stderr, "addWithCuda failed!"); return 1; }

    return 0;
}

