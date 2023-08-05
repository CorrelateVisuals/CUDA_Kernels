#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

__global__ void getKernelDetails(int* array, int key) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    switch (key) {
        case 0:
            array[i] = blockDim.x;
            break;
        case 1:
            array[i] = threadIdx.x;
            break;
        case 2:
            array[i] = blockIdx.x;
            break;
        case 3:
            array[i] = i;
            break;
    }
}

void printResults(int* array, int arraySize, const std::string& source) {
    std::cout << source;
    for (size_t i = 0; i < arraySize; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    const dim3 gridSizeX = { 3, 1, 1 };
    const dim3 numThreads = { 4, 1, 1 };
    const int arraySize = gridSizeX.x * numThreads.x;

    int* h_blockDimensions = new int[arraySize];
    int* h_localThreads = new int[arraySize];
    int* h_blockIndices = new int[arraySize];
    int* h_threadsIndices = new int[arraySize];

    int* d_blockDimensions;
    int* d_arrayLocalThread;
    int* d_blockIndices;
    int* d_threadsIndices;

    // Block Dimensions
    cudaMalloc((void**)&d_blockDimensions, arraySize * sizeof(int));
    cudaMemcpy(d_blockDimensions, h_blockDimensions, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    getKernelDetails<<<gridSizeX, numThreads>>>(d_blockDimensions, 0);

    // Local Thread Index
    cudaMalloc((void**)&d_arrayLocalThread, arraySize * sizeof(int));
    cudaMemcpy(d_arrayLocalThread, h_localThreads, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    getKernelDetails<<<gridSizeX, numThreads>>>(d_arrayLocalThread, 1);

    // Block Index
    cudaMalloc((void**)&d_blockIndices, arraySize * sizeof(int));
    cudaMemcpy(d_blockIndices, h_blockIndices, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    getKernelDetails<<<gridSizeX, numThreads>>>(d_blockIndices, 2);

    // Thread Index
    cudaMalloc((void**)&d_threadsIndices, arraySize * sizeof(int));
    cudaMemcpy(d_threadsIndices, h_threadsIndices, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    getKernelDetails<<<gridSizeX, numThreads>>>(d_threadsIndices, 3);

    cudaMemcpy(h_blockDimensions, d_blockDimensions, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_localThreads, d_arrayLocalThread, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blockIndices, d_blockIndices, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_threadsIndices, d_threadsIndices, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_blockDimensions);
    cudaFree(d_arrayLocalThread);
    cudaFree(d_blockIndices);
    cudaFree(d_threadsIndices);

    cudaDeviceSynchronize();

    printResults(h_blockDimensions, arraySize, "Kernel block dimensions: ");
    printResults(h_localThreads, arraySize, "Kernel local thread index: ");
    printResults(h_blockIndices, arraySize, "Kernel block index: ");
    printResults(h_threadsIndices, arraySize, "Kernel thread index: ");

    delete[] h_blockDimensions;
    delete[] h_localThreads;
    delete[] h_blockIndices;
    delete[] h_threadsIndices;

    return 0;
}
