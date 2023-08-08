#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

using namespace std;

void printDeviceProperties(int deviceIndex) {
    cudaDeviceProp properties;
    cudaError_t cudaStatus = cudaGetDeviceProperties(&properties, deviceIndex);
    if (cudaStatus != cudaSuccess) {
        cerr << "Error getting device properties" << endl;
        return;
    }

    cout << "Device Index: " << deviceIndex << endl;
    cout << "Device Name: " << properties.name << endl;
    cout << "Total Global Memory: " << properties.totalGlobalMem << " bytes" << endl;
    cout << "Shared Memory Per Block: " << properties.sharedMemPerBlock << " bytes" << endl;
    cout << "Registers Per Block: " << properties.regsPerBlock << endl;
    cout << "Warp Size: " << properties.warpSize << endl;
    cout << "Memory Pitch: " << properties.memPitch << " bytes" << endl;
    cout << "Max Threads Per Block: " << properties.maxThreadsPerBlock << endl;
    cout << "Max Threads Dim: [" << properties.maxThreadsDim[0] << ", " << properties.maxThreadsDim[1] << ", " << properties.maxThreadsDim[2] << "]" << endl;
    cout << "Max Grid Size: [" << properties.maxGridSize[0] << ", " << properties.maxGridSize[1] << ", " << properties.maxGridSize[2] << "]" << endl;
    cout << "Total Constant Memory: " << properties.totalConstMem << " bytes" << endl;
    cout << "Major Compute Capability: " << properties.major << endl;
    cout << "Minor Compute Capability: " << properties.minor << endl;
    cout << "Clock Rate: " << properties.clockRate << " kHz" << endl;
    cout << "Texture Alignment: " << properties.textureAlignment << " bytes" << endl;
    cout << "Device Overlap: " << properties.deviceOverlap << endl;
    cout << "Multiprocessor Count: " << properties.multiProcessorCount << endl;
    cout << "Kernel Execution Timeout Enabled: " << properties.kernelExecTimeoutEnabled << endl;
    cout << "Integrated GPU: " << properties.integrated << endl;
    cout << "Can Map Host Memory: " << properties.canMapHostMemory << endl;
    cout << "Compute Mode: " << properties.computeMode << endl;
    cout << "Max Texture 1D: " << properties.maxTexture1D << endl;
    cout << "Max Texture 2D: [" << properties.maxTexture2D[0] << ", " << properties.maxTexture2D[1] << "]" << endl;
    cout << "Max Texture 3D: [" << properties.maxTexture3D[0] << ", " << properties.maxTexture3D[1] << ", " << properties.maxTexture3D[2] << "]" << endl;
    cout << "Concurrent Kernels: " << properties.concurrentKernels << endl;
}

int main() {
    int deviceCount;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess) {
        cerr << "Error getting device count" << endl;
        return 1;
    }

    for (int i = 0; i < deviceCount; i++) {
        printDeviceProperties(i);
        cout << endl;
    }

    return 0;
}
