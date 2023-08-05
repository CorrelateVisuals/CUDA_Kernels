#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>

#include "STB_Image_Load.h"
#include "STB_Image_Write.h"

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cout << "Usage: Image Color Manipulation <filename>" << std::endl;
        return -1;
    }

    int width, height, componentCount, channels = 4;
    std::cout << "Loading png file ... ";
    unsigned char* imageData = stbi_load(argv[1], &width, &height, &componentCount, channels);
    if (!imageData) {
        std::cout << "Failed to open \"" << argv[1] << "\"";
        return -1;
    }
    std::cout << "Done" << std::endl;

    if (width % 32 || height % 32) {
        std::cout << "Width or height is not devidible by 32; leaked memory of imageData";
        return -1;
    }

    std::string fileNameOut = argv[1];
    int strideBytes = sizeof(int) * width;

    std::cout << "Writing png to disk ... ";
    std::string baseFileName = fileNameOut.substr(0, fileNameOut.find_last_of("."));
    std::string newFileName = baseFileName + "_gray.jpg";
    stbi_write_png(newFileName.c_str(), width, height, channels, imageData, strideBytes);
    std::cout << "Done " << fileNameOut << " saved" << std::endl;

    stbi_image_free(imageData);

    return 0;
}
