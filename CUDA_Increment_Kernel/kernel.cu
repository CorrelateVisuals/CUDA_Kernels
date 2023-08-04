#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

void incrementCPU(int* a, int b, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = a[i] + 10;
    }
}

__global__ void incrementGPU(int* a, int b, int N) {
    int i = threadIdx.x;
    if (i < N) {
        a[i] = a[i] + b;
    };
}

void printResults(int* a, int N, const std::string& source) {
    std::cout << "Incremented on " << source << ": { ";
    for (int i = 0; i < N; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << "}" << std::endl;
}

int main() {
    const int N = 10;
    int stepSize = 10;
    int h_initialValues[N] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    // CPU execution
    incrementCPU(h_initialValues, stepSize, N);
    printResults(h_initialValues, N, "CPU");

    // GPU execution
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(N, 1, 1);
    int* d_initialValues;

    cudaMalloc((void**)&d_initialValues, N * sizeof(int));
    cudaMemcpy(d_initialValues, h_initialValues, N * sizeof(int), cudaMemcpyHostToDevice);

    incrementGPU << < gridSize, blockSize >> > (d_initialValues, stepSize, N);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_initialValues);
        return 1;
    }

    cudaMemcpy(h_initialValues, d_initialValues, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_initialValues);

    printResults(h_initialValues, N, "GPU");

    return 0;
}
