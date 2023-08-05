#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

void incrementCPU(int* a, int arraySize) {
    for (int i = 0; i < arraySize; i++) {
        a[i] = a[i] + 10;
    }
}

__global__ void incrementGPU(int* a, int arraySize) {
    int i = threadIdx.x;
    int stepSize = 10;
    if (i < arraySize) {
        a[i] = a[i] + stepSize;
    };
}

void printResults(int* a, int arraySize, const std::string& source) {
    std::cout << "Incremented on " << source << ": { ";
    for (int i = 0; i < arraySize; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << "}" << std::endl;
}

int main() {
    const int arraySize = 10;
    int h_initialValues[arraySize] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    // CPU execution
    incrementCPU(h_initialValues, arraySize);
    printResults(h_initialValues, arraySize, "CPU");

    // GPU execution
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(arraySize, 1, 1);
    int* d_initialValues;

    cudaMalloc((void**)&d_initialValues, arraySize * sizeof(int));
    cudaMemcpy(d_initialValues, h_initialValues, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    incrementGPU<<< gridSize, blockSize >>>(d_initialValues, arraySize);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_initialValues);
        return 1;
    }

    cudaMemcpy(h_initialValues, d_initialValues, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_initialValues);

    printResults(h_initialValues, arraySize, "GPU");

    return 0;
}
