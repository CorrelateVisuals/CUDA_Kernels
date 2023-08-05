#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <random>
#include <cassert>
#include <iomanip>
#include <vector>
#include <ctime>

const size_t WORKING_SET_COUNT = 256 * 1024;

struct Vector {
	float x, y, z, w;
};

__global__ void addVectorGPU(const Vector* arrayA, const Vector* arrayB, Vector* arrayResult) {
	size_t i = threadIdx.x + static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x);

	arrayResult[i].x = arrayA[i].x + arrayB[i].x;
	arrayResult[i].y = arrayA[i].y + arrayB[i].y;
	arrayResult[i].z = arrayA[i].z + arrayB[i].z;
	arrayResult[i].w = arrayA[i].w + arrayB[i].w;
}

int main() {
	std::vector<Vector> h_vectorArrayA(WORKING_SET_COUNT);
	std::vector<Vector> h_vectorArrayB(WORKING_SET_COUNT);
	std::vector<Vector> h_vectorArrayResult(WORKING_SET_COUNT);

	assert(h_vectorArrayA && h_vectorArrayB && h_vectorArrayResult && "Memory allocation failed for working set");

	std::srand(std::time(nullptr));
	for (size_t i = 0; i < WORKING_SET_COUNT; i++) {
		h_vectorArrayA[i].x = 1.0f / (std::rand() % 200 + 1);
		h_vectorArrayA[i].y = 1.0f / (std::rand() % 200 + 1);
		h_vectorArrayA[i].z = 1.0f / (std::rand() % 200 + 1);
		h_vectorArrayA[i].w = 1.0f / (std::rand() % 200 + 1);

		h_vectorArrayB[i].x = 1.0f / (std::rand() % 200 + 1);
		h_vectorArrayB[i].y = 1.0f / (std::rand() % 200 + 1);
		h_vectorArrayB[i].z = 1.0f / (std::rand() % 200 + 1);
		h_vectorArrayB[i].w = 1.0f / (std::rand() % 200 + 1);
	}

	Vector* d_vectorArrayA = nullptr, * d_vectorArrayB = nullptr, * d_vectorArrayResult = nullptr;

	cudaMalloc(&d_vectorArrayA, sizeof(Vector) * WORKING_SET_COUNT);
	cudaMalloc(&d_vectorArrayB, sizeof(Vector) * WORKING_SET_COUNT);
	cudaMalloc(&d_vectorArrayResult, sizeof(Vector) * WORKING_SET_COUNT);
	assert(d_vectorArrayA && d_vectorArrayB && d_vectorArrayResult && "Cuda memory allocation failed for working set");

	cudaMemcpy(d_vectorArrayA, h_vectorArrayA.data(), sizeof(Vector) * WORKING_SET_COUNT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vectorArrayB, h_vectorArrayB.data(), sizeof(Vector) * WORKING_SET_COUNT, cudaMemcpyHostToDevice);

	const size_t blockSize = 256;
	const size_t blockCount = WORKING_SET_COUNT / blockSize;

	addVectorGPU<<<blockSize, blockCount>>>(d_vectorArrayA, d_vectorArrayB, d_vectorArrayResult);
	cudaDeviceSynchronize();

	cudaMemcpy(h_vectorArrayResult.data(), d_vectorArrayResult, sizeof(Vector) * WORKING_SET_COUNT, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < 5; i++) {
		std::cout << std::setw(8) << std::fixed 
			<<   "X = " << h_vectorArrayResult[i].x 
			<< "\tY = " << h_vectorArrayResult[i].y 
			<< "\tZ = " << h_vectorArrayResult[i].z 
			<< "\tW = " << h_vectorArrayResult[i].w << std::endl;
	}

	cudaFree(d_vectorArrayA);
	cudaFree(d_vectorArrayB);
	cudaFree(d_vectorArrayResult);
}