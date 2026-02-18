# Thread Indices Kernel

This CUDA program explores the hierarchical organization of threads in GPU computing by launching kernels with various configurations and displaying detailed thread information. Understanding thread organization is fundamental to writing efficient CUDA programs.

## Overview

The program demonstrates CUDA's three-level thread hierarchy:
1. **Grid**: The entire collection of threads launched by a kernel
2. **Block**: Groups of threads that can cooperate and share memory
3. **Thread**: Individual execution unit

## Key Concepts

### Thread Hierarchy

CUDA organizes threads in a hierarchical structure:
- **threadIdx**: Thread's position within its block (x, y, z coordinates)
- **blockIdx**: Block's position within the grid (x, y, z coordinates)
- **blockDim**: Number of threads per block (x, y, z dimensions)
- **gridDim**: Number of blocks in the grid (x, y, z dimensions)

### Global Thread Index

To compute a unique global index for each thread:
```cuda
int idx = threadIdx.x + blockIdx.x * blockDim.x;
```

This formula maps the 2D thread organization (blocks and threads) to a 1D array index.

## What This Program Shows

The kernel execution demonstrates:
- How threads are distributed across blocks
- Thread and block indexing in different configurations
- The relationship between block dimensions and thread indices
- Parallel execution coordination across the GPU grid

## Learning Outcomes

By running this program with different configurations, you'll understand:
- How to choose appropriate block and grid sizes
- The limitations on threads per block (typically max 1024)
- How thread organization affects memory access patterns
- The importance of thread indexing for data processing

## Usage

Compile and run to see thread organization details:

```bash
nvcc kernel.cu -o thread_indices
./thread_indices
```

The program launches kernels with multiple configurations, displaying how threads are organized and indexed within blocks and the overall grid structure.