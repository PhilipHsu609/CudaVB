#include "my_cuda_lib.h"
#include "helper_cuda.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <algorithm>
#include <iostream>

void thinningCpu(const uint8_t *src, uint8_t *dst, int width, int height);
void thinningCuda(const uint8_t *devSrc, uint8_t *devDst, int width, int height);

std::vector<uint8_t> neighbor(const uint8_t *src, int y, int x, int width);
int ZSA(const std::vector<uint8_t> &n);
int ZSB(const std::vector<uint8_t> &n);
int ZSsub1(const uint8_t *src, uint8_t *dst, int width, int height);
int ZSsub2(const uint8_t *src, uint8_t *dst, int width, int height);

__device__ void neighbor(const uint8_t *src, int y, int x, int width, uint8_t n[8]);
__device__ int ZSA(const uint8_t n[8]);
__device__ int ZSB(const uint8_t n[8]);
__global__ void ZSsub1Kernel(const uint8_t *src, uint8_t *dst, int width, int height, int *removed);
__global__ void ZSsub2Kernel(const uint8_t *src, uint8_t *dst, int width, int height, int *revmoed);

__device__ void neighbor(const uint8_t *src, int y, int x, int width, uint8_t n[8]) {
	/*
		P9 P2 P3
		P8 P1 P4
		P7 P6 P5

		P1: [y, x]
		n: [P2, P3, P4, P5, P6, P7, P8, P9]
	*/
	n[0] = src[(y - 1) * width + x];     // P2
	n[1] = src[(y - 1) * width + x + 1]; // P3
	n[2] = src[y * width + x + 1];	     // P4
	n[3] = src[(y + 1) * width + x + 1]; // P5
	n[4] = src[(y + 1) * width + x];     // P6
	n[5] = src[(y + 1) * width + x - 1]; // P7
	n[6] = src[y * width + x - 1];	     // P8
	n[7] = src[(y - 1) * width + x - 1]; // P9
}

__device__ int ZSA(const uint8_t n[8]) {
	int ret{ 0 };
	for (int i = 0; i < sizeof(n) - 1; i++) {
		if (n[i] == 0 && n[i + 1] == 0xFF) {
			ret++;
		}
	}
	ret += (n[7] == 0 && n[0] == 0xFF) ? 1 : 0;
	return ret;
}

__device__ int ZSB(const uint8_t n[8]) {
	int ret{ 0 };
	for (int i = 0; i < sizeof(n); i++) {
		ret += (n[i] == 0xFF) ? 1 : 0;
	}
	return ret;
}

__global__ void ZSsub1Kernel(const uint8_t *src, uint8_t *dst, int width, int height, int *removed) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y >= height || x >= width) {
		return;
	}

	if (y == 0 || x == 0 || y == height - 1 || x == width - 1 || src[y * width + x] == 0) {
		dst[y * width + x] = 0;
		return;
	}

	uint8_t n[8];
	neighbor(src, y, x, width, n);
	int A{ ZSA(n) };
	int B{ ZSB(n) };

	/*
		a. 2 <= B <= 6
		b. A = 1
		c. P2 * P4 * P6 = 0
		d. P4 * P6 * P8 = 0
	*/
	if (A == 1 && B >= 2 && B <= 6 && int(n[0]) * n[2] * n[4] == 0 && int(n[2]) * n[4] * n[6] == 0) {
		dst[y * width + x] = 0;
		(*removed)++;	// use atomicAdd if you care about the exact number
	} else {
		dst[y * width + x] = src[y * width + x];
	}
}
__global__ void ZSsub2Kernel(const uint8_t *src, uint8_t *dst, int width, int height, int *removed) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y >= height || x >= width) {
		return;
	}

	if (y == 0 || x == 0 || y == height - 1 || x == width - 1 || src[y * width + x] == 0) {
		dst[y * width + x] = 0;
		return;
	}

	uint8_t n[8];
	neighbor(src, y, x, width, n);
	int A{ ZSA(n) };
	int B{ ZSB(n) };

	/*
		a. 2 <= B <= 6
		b. A = 1
		c. P2 * P4 * P8 = 0
		d. P2 * P6 * P8 = 0
	*/
	if (A == 1 && B >= 2 && B <= 6 && int(n[0]) * n[2] * n[6] == 0 && int(n[0]) * n[4] * n[6] == 0) {
		dst[y * width + x] = 0;
		(*removed)++;	// use atomicAdd if you care about the exact number
	} else {
		dst[y * width + x] = src[y * width + x];
	}
}

void thinningCuda(const uint8_t *devSrc, uint8_t *devDst, int width, int height) {
	// 1. copy src to dst
	// 2. allocate temp buffer
	// 3. ZSsub1(dst, tmp)
	// 4. ZSsub2(tmp, dst)

	uint8_t *devTmp{};
	checkCudaErrors(cudaMemcpy(devDst, devSrc, width * height, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMalloc(&devTmp, width * height));

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(divUp(width, threadsPerBlock.x), divUp(height, threadsPerBlock.y));

	int removed{ 1 }, *devRemoved{};
	checkCudaErrors(cudaMalloc(&devRemoved, sizeof(int)));

	while (removed != 0) {
		checkCudaErrors(cudaMemset(devRemoved, 0, sizeof(int)));
		ZSsub1Kernel<<<numBlocks, threadsPerBlock>>>(devDst, devTmp, width, height, devRemoved);
		ZSsub2Kernel<<<numBlocks, threadsPerBlock>>>(devTmp, devDst, width, height, devRemoved);
		checkCudaErrors(cudaMemcpy(&removed, devRemoved, sizeof(int), cudaMemcpyDeviceToHost));
	}

	checkCudaErrors(cudaFree(devRemoved));
	checkCudaErrors(cudaFree(devTmp));
}

int ZSA(const std::vector<uint8_t> &n) {
	int ret{ 0 };
	for (int i = 0; i < n.size() - 1; i++) {
		if (n[i] == 0 && n[i + 1] == 0xFF) {
			ret++;
		}
	}
	ret += (n.back() == 0 && n.front() == 0xFF) ? 1 : 0;
	return ret;
}

int ZSB(const std::vector<uint8_t> &n) {
	return std::count(n.begin(), n.end(), 0xFF);
}

std::vector<uint8_t> neighbor(const uint8_t *src, int y, int x, int width) {
	/*
		P9 P2 P3
		P8 P1 P4
		P7 P6 P5

		P1: [y, x]
		ret: [P2, P3, P4, P5, P6, P7, P8, P9]
	*/
	std::vector<uint8_t> ret(8);
	ret[0] = src[(y - 1) * width + x];     // P2
	ret[1] = src[(y - 1) * width + x + 1]; // P3
	ret[2] = src[y * width + x + 1];	   // P4
	ret[3] = src[(y + 1) * width + x + 1]; // P5
	ret[4] = src[(y + 1) * width + x];	   // P6
	ret[5] = src[(y + 1) * width + x - 1]; // P7
	ret[6] = src[y * width + x - 1];	   // P8
	ret[7] = src[(y - 1) * width + x - 1]; // P9
	return ret;
}

int ZSsub1(const uint8_t *src, uint8_t *dst, int width, int height) {
	int removed{ 0 };
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			if (y == 0 || x == 0 || y == height - 1 || x == width - 1 || src[y * width + x] == 0) {
				dst[y * width + x] = 0;
				continue;
			}

			auto n{ neighbor(src, y, x, width) };
			int A{ ZSA(n) };
			int B{ ZSB(n) };

			/*
				a. 2 <= B <= 6
				b. A = 1
				c. P2 * P4 * P6 = 0
				d. P4 * P6 * P8 = 0
			*/
			if (A == 1 && B >= 2 && B <= 6 && int(n[0]) * n[2] * n[4] == 0 && int(n[2]) * n[4] * n[6] == 0) {
				dst[y * width + x] = 0;
				removed++;
			} else {
				dst[y * width + x] = src[y * width + x];
			}
		}
	}
	return removed;
}

int ZSsub2(const uint8_t *src, uint8_t *dst, int width, int height) {
	int removed{ 0 };
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (y == 0 || x == 0 || y == height - 1 || x == width - 1 || src[y * width + x] == 0) {
				dst[y * width + x] = 0;
				continue;
			}

			auto n{ neighbor(src, y, x, width) };
			int A{ ZSA(n) };
			int B{ ZSB(n) };

			/*
				a. 2 <= B <= 6
				b. A = 1
				c. P2 * P4 * P8 = 0
				d. P2 * P6 * P8 = 0
			*/
			if (A == 1 && B >= 2 && B <= 6 && int(n[0]) * n[2] * n[6] == 0 && int(n[0]) * n[4] * n[6] == 0) {
				dst[y * width + x] = 0;
				removed++;
			} else {
				dst[y * width + x] = src[y * width + x];
			}
		}
	}
	return removed;
}

void thinningCpu(const uint8_t *src, uint8_t *dst, int width, int height) {
	// 1. copy src to dst
	// 2. allocate temp buffer
	// 3. ZSsub1(dst, tmp)
	// 4. ZSsub2(tmp, dst)
	//
	// Ref: http://agcggs680.pbworks.com/f/Zhan-Suen_algorithm.pdf

	std::copy(src, src + width * height, dst);
	std::vector<uint8_t> tmp_(width * height);
	uint8_t *tmp = tmp_.data();

	int removed{ 1 };
	while (removed != 0) {
		removed = ZSsub1(dst, tmp, width, height);
		removed += ZSsub2(tmp, dst, width, height);
	}
}
