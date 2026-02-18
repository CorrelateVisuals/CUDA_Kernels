# Image Color Manipulation Kernel

This CUDA program performs GPU-accelerated grayscale conversion on JPEG images using weighted RGB channel averaging. It also generates a thread visualization image showing how different GPU threads process different regions of the image.

## Overview

The program demonstrates GPU-accelerated image processing by:
1. Loading a JPEG image using the STB image library
2. Transferring image data to GPU memory
3. Executing a CUDA kernel for parallel grayscale conversion
4. Generating two output images:
   - **[filename]_grey.jpg**: Grayscale version of the input
   - **[filename]_threads.jpg**: Visualization of thread assignments

## Grayscale Conversion Algorithm

The kernel uses the standard luminosity method for grayscale conversion:
```
gray_value = 0.299 * R + 0.587 * G + 0.114 * B
```

These weights account for human eye sensitivity, where green contributes most to perceived brightness.

## Thread Visualization

The thread visualization image shows:
- How the image is divided among different GPU threads
- Which thread block processes each region
- The parallel nature of image processing on the GPU

This visualization helps understand how CUDA distributes work across the GPU's parallel processing units.

## Example

### Original Image
![Original Image](https://github.com/CorrelateVisuals/Nvidea_CUDA/blob/main/Image_Color_Manipulation_Kernel/galaxy.jpg?raw=true)

**Image Details:**
- **Object**: Stephan's Quintet
- **Telescope**: Grantecan / Nasmyth-B
- **Instrument**: OSIRIS
- **Filters**: G (481nm), R (641nm), I (770nm), Z (970nm), OS657 (657nm, FWHM 35nm)
- **Color Mapping**: Blue (G), Green (Blue+Red), Red (R+I+Z), Yellow (OS657)
- **Exposure**: 4 × 30 secs (G, R, I, Z), 4 × 90 secs (OS657)
- **Field of View**: Approx. 5' × 6'
- **Orientation**: North is up, East is left
- **Position**: RA(J2000.0) = 22h35m57s, Dec(J2000.0) = 33°57'36"
- **Image Processing**: Daniel López/IAC

Original © GRANTECAN S.A.

### Grayscale Output
![Grayscale](https://github.com/CorrelateVisuals/Nvidea_CUDA/blob/main/Image_Color_Manipulation_Kernel/galaxy_grey.jpg?raw=true)

Grayscale © GRANTECAN S.A.

### Thread Visualization
![Thread Visualization](https://github.com/CorrelateVisuals/Nvidea_CUDA/blob/main/Image_Color_Manipulation_Kernel/galaxy_threads.jpg?raw=true)

This visualization displays which thread processes each pixel, helping visualize the parallel execution pattern.

## Usage

Compile and run the program:

```bash
nvcc kernel.cu STB_Image_Load.cpp STB_Image_Write.cpp -o image_processing
./image_processing input_image.jpg
```

The program will generate:
- `input_image_grey.jpg` - Grayscale version
- `input_image_threads.jpg` - Thread visualization

## Key Concepts

- **Parallel Image Processing**: Each pixel is processed independently by a different thread
- **2D Thread Indexing**: Image coordinates map to 2D thread blocks
- **Memory Coalescing**: Efficient memory access patterns for image data
- **Real-world Application**: Demonstrates practical GPU acceleration for image processing

## Dependencies

- **STB Image**: Header-only image loading/writing library (included)
- **CUDA Toolkit**: For compilation and execution

