# Vector Addition
This code demonstrates how to use CUDA to perform parallel vector addition on the GPU. It creates two arrays of 4D vectors, transfers them to the GPU, and launches a CUDA kernel to perform the addition in parallel. The results are transferred back to the CPU, and the first five elements of the resulting vector array are printed. The parallel computation significantly accelerates the vector addition, as shown in the NVIDEA Nsight Systems 2021.3.3 performance analysis results. Just like with the increment the biggest overhead is copying the memory from host to device and back. Something to be aware of while optimizing your computations.

![alt text](https://github.com/CorrelateVisuals/Nvidea_CUDA/blob/main/Vector_Addition_Kernel/Nvidea_Nsight.PNG?raw=true)
