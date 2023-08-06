# Increment Kernel
The code demonstrates how to use CUDA, a parallel computing platform, to increment elements of an array both on the CPU and the GPU. It showcases the concept of parallelism by launching a 
kernel on the GPU that performs the increment operation concurrently for multiple array elements. This approach significantly speeds up the computation compared to the sequential CPU execution. 
Although the proces of copying memory costs more time than a simpel calculation like this. However, if scaled up you will see its benefits.