# Image Color Manipulation (greyscale)
This CUDA kernel performs grayscale conversion on a jpg image, using weighted averages of RGB channels to determine pixel intensity. It loads the image, transfers data to the GPU, executes the CUDA kernel for grayscale conversion, and saves the resulting image back to disk with "_grey" appended to the filename.

Original © GRANTECAN S.A.
![Image](https://github.com/CorrelateVisuals/Nvidea_CUDA/blob/main/Image_Color_Manipulation_Kernel/galaxy.jpg?raw=true)
Greyscale © GRANTECAN S.A.
![Image](https://github.com/CorrelateVisuals/Nvidea_CUDA/blob/main/Image_Color_Manipulation_Kernel/galaxy_grey.jpg?raw=true)
