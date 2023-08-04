﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

void incrementCPU(int* a, int b, int N) {
    std::cout << "Incremented on CPU: { ";
    for (int i = 0; i < N; i++) {
        a[i] = a[i] + 10;
        std::cout << a[i] << " ";
    };
    std::cout << "}" << std::endl;
}

__global__ void incrementGPU(int* a, int b, int N) {
    int i = threadIdx.x;
    if (i < N) {
        a[i] = a[i] + b;
    };
}

int main() {
    const int N = 10;
    int incrementBy = 10;
    int h_initialValues[N] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    // CPU execution
    incrementCPU(h_initialValues, incrementBy, N);

    // GPU execution
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(N, 1, 1);
    int* d_initialValues;

    cudaMalloc((void**)&d_initialValues, N * sizeof(int));
    cudaMemcpy(d_initialValues, h_initialValues, N * sizeof(int), cudaMemcpyHostToDevice);
    incrementGPU << < gridSize, blockSize >> > (d_initialValues, incrementBy, N);

    cudaMemcpy(h_initialValues, d_initialValues, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_initialValues);

    std::cout << "Incremented on GPU: { ";
    for (int i = 0; i < N; i++) {
        std::cout << h_initialValues[i] << " ";
    }
    std::cout << "}" << std::endl;

    return 0;
}
