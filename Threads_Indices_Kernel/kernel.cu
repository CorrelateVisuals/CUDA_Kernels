#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

__global__ void getKernelBlockDimensions(int* array) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    array[i] = blockDim.x;
}

__global__ void getKernelLocalhreadIndex(int* array) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    array[i] = threadIdx.x;
}

__global__ void getKernelBlockIndex(int* array) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    array[i] = blockIdx.x;
}

__global__ void getKernelThreadIndex(int* array) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    array[i] = i;
}

void printResults(int* array, int arraySize, const std::string& source) {
    std::cout << source;
    for (int i = 0; i < arraySize; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    const dim3 gridSizex = { 3, 1, 1 };
    const dim3 numThreads = { 4, 1, 1 };
    const int arraySize = gridSizex.x * numThreads.x;

    int* h_array_block_dims = new int[arraySize];
    int* h_array_local_thread = new int[arraySize];
    int* h_array_block_index = new int[arraySize];
    int* h_array_thread_index = new int[arraySize];

    int* d_array_block_dims;
    int* d_array_local_thread;
    int* d_array_block_index;
    int* d_array_thread_index;

    // Block Dimensions
    cudaMalloc((void**)&d_array_block_dims, arraySize * sizeof(int));
    cudaMemcpy(d_array_block_dims, h_array_block_dims, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    getKernelBlockDimensions << <gridSizex, numThreads >> > (d_array_block_dims);

    // Local Thread Index
    cudaMalloc((void**)&d_array_local_thread, arraySize * sizeof(int));
    cudaMemcpy(d_array_local_thread, h_array_local_thread, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    getKernelLocalhreadIndex << <gridSizex, numThreads >> > (d_array_local_thread);

    // Block Index
    cudaMalloc((void**)&d_array_block_index, arraySize * sizeof(int));
    cudaMemcpy(d_array_block_index, h_array_block_index, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    getKernelBlockIndex << <gridSizex, numThreads >> > (d_array_block_index);

    // Thread Index
    cudaMalloc((void**)&d_array_thread_index, arraySize * sizeof(int));
    cudaMemcpy(d_array_thread_index, h_array_thread_index, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    getKernelThreadIndex << <gridSizex, numThreads >> > (d_array_thread_index);

    cudaMemcpy(h_array_block_dims, d_array_block_dims, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_array_local_thread, d_array_local_thread, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_array_block_index, d_array_block_index, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_array_thread_index, d_array_thread_index, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_array_block_dims);
    cudaFree(d_array_local_thread);
    cudaFree(d_array_block_index);
    cudaFree(d_array_thread_index);

    cudaDeviceSynchronize();

    printResults(h_array_block_dims, arraySize, "Kernel block dimensions: ");
    printResults(h_array_local_thread, arraySize, "Kernel local thread index: ");
    printResults(h_array_block_index, arraySize, "Kernel block index: ");
    printResults(h_array_thread_index, arraySize, "Kernel thread index: ");

    delete[] h_array_block_dims;
    delete[] h_array_local_thread;
    delete[] h_array_block_index;
    delete[] h_array_thread_index;

    return 0;
}
