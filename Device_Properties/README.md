# Device Properties

This CUDA program queries and displays detailed properties of all CUDA-capable GPUs in your system. It's useful for understanding your hardware capabilities and determining optimal kernel configuration parameters.

## Overview

The program uses CUDA's device query API to retrieve comprehensive information about each GPU, including memory specifications, compute capabilities, and hardware features. This information is essential for:

- Understanding hardware limitations
- Optimizing kernel configurations
- Ensuring compatibility across different GPU models
- Debugging CUDA applications

## Usage

Compile and run the program to see your GPU properties:

```bash
nvcc kernel.cu -o device_properties
./device_properties
```

## Sample Output

Example output from an NVIDIA GeForce GTX 1070:

```
Device Index: 0
Device Name: NVIDIA GeForce GTX 1070
Total Global Memory: 8589737984 bytes
Shared Memory Per Block: 49152 bytes
Registers Per Block: 65536
Warp Size: 32
Memory Pitch: 2147483647 bytes
Max Threads Per Block: 1024
Max Threads Dim: [1024, 1024, 64]
Max Grid Size: [2147483647, 65535, 65535]
Total Constant Memory: 65536 bytes
Major Compute Capability: 6
Minor Compute Capability: 1
Clock Rate: 1645000 kHz
Texture Alignment: 512 bytes
Device Overlap: 1
Multiprocessor Count: 16
Kernel Execution Timeout Enabled: 1
Integrated GPU: 0
Can Map Host Memory: 1
Compute Mode: 0
Max Texture 1D: 131072
Max Texture 2D: [131072, 65536]
Max Texture 3D: [16384, 16384, 16384]
Concurrent Kernels: 1
```

## Key Properties Explained

- **Total Global Memory**: Available GPU memory for allocations
- **Compute Capability**: GPU architecture version (major.minor)
- **Multiprocessor Count**: Number of streaming multiprocessors (SMs)
- **Max Threads Per Block**: Maximum threads in a single block
- **Warp Size**: Number of threads executed together (typically 32)
- **Shared Memory Per Block**: Fast on-chip memory per block
