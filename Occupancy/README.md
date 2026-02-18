# Occupancy Calculator

This CUDA kernel demonstrates how to calculate and optimize GPU occupancy using CUDA's occupancy API. Understanding occupancy is crucial for maximizing GPU performance and resource utilization.

## Overview

The program calculates the occupancy of a simple kernel by determining how many active blocks can run per multiprocessor (SM). This information helps developers optimize their kernel configurations for better performance.

## What is Occupancy?

Occupancy is the ratio of active warps to the maximum number of warps supported on a multiprocessor. Higher occupancy doesn't always guarantee better performance, but it provides more opportunities to hide latency through context switching.

## Code Explanation

The kernel performs simple element-wise multiplication:
```cuda
__global__ void MyKernel(int* d, int* a, int* b)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx];
}
```

The host code uses `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` to determine:
- Maximum number of active blocks per SM for a given block size
- Resulting occupancy as a percentage of maximum possible warps

## Key Concepts

- **Block Size**: Number of threads per block (64 in this example)
- **Active Blocks**: Maximum blocks that can execute simultaneously on one SM
- **Active Warps**: Number of warps that can be active (numBlocks × blockSize / warpSize)
- **Occupancy**: Percentage of maximum warps that are active

## Sample Output

```
Occupancy: 100%
```

This indicates the kernel configuration achieves maximum theoretical occupancy on the device.

## Usage

Compile and run to see the occupancy percentage for your GPU:
```bash
nvcc kernel.cu -o occupancy
./occupancy
```

The program will display the occupancy percentage based on your GPU's capabilities.
