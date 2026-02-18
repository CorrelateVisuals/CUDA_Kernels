# CUDA Kernels

A comprehensive collection of NVIDIA CUDA kernel examples exploring parallel computation concepts and GPU programming techniques using CUDA 11.5. This repository provides practical implementations demonstrating various aspects of CUDA programming, from basic device properties to advanced image processing operations.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Available Kernels](#available-kernels)
- [Building and Running](#building-and-running)
- [License](#license)

## Overview

This repository contains multiple CUDA kernel implementations that showcase different aspects of parallel computing on NVIDIA GPUs. Each kernel is designed to demonstrate specific CUDA programming concepts and best practices.

## Prerequisites

- **NVIDIA GPU**: CUDA-capable GPU with compute capability 3.0 or higher
- **CUDA Toolkit**: CUDA 11.5 or compatible version
- **Visual Studio**: Visual Studio 2019 or later (for Windows development)
- **NVIDIA Nsight Systems**: Optional, for performance profiling and analysis

## Available Kernels

### [Device Properties](Device_Properties/)
Retrieves and displays all properties of CUDA-capable GPUs on your system, including memory specifications, compute capabilities, and hardware configuration.

### [Increment Kernel](Increment_Kernel/)
Demonstrates basic parallel computation by incrementing array elements on both CPU and GPU, showcasing the performance differences between sequential and parallel execution.

### [Threads and Indices](Threads_Indices_Kernel/)
Explores CUDA kernel execution details by displaying thread organization, block dimensions, and grid structure. Provides insights into how CUDA manages and coordinates thread execution.

### [Vector Addition](Vector_Addition_Kernel/)
Implements parallel vector addition on the GPU using 4D vectors. Includes performance analysis using NVIDIA Nsight Systems, demonstrating the overhead of memory transfers and the benefits of parallel computation.

### [Image Color Manipulation](Image_Color_Manipulation_Kernel/)
Performs grayscale conversion on JPEG images using weighted RGB channel averages. Demonstrates GPU-accelerated image processing and includes thread visualization capabilities.

### [Occupancy Calculator](Occupancy/)
Calculates and displays GPU occupancy metrics to help optimize kernel performance by understanding resource utilization and thread organization.

## Building and Running

### Windows (Visual Studio)

1. Open the solution file (`.sln`) in the desired kernel directory
2. Ensure CUDA Toolkit is properly installed and configured in Visual Studio
3. Build the project using Visual Studio (F7 or Build → Build Solution)
4. Run the executable from the output directory

### Command Line (nvcc)

For individual kernels, you can compile using nvcc:

```bash
nvcc kernel.cu -o output_name
./output_name
```

### Performance Profiling

To profile kernels using NVIDIA Nsight Systems:

```bash
nsys profile ./your_executable
```

## License

This project is licensed under CC0 1.0 Universal. See the [LICENSE](LICENSE) file for details.
