#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "my_cuda_lib.h"
#include "cooperative_groups.h"
#include "utils.h"

#include <cmath>
#include <vector>

#define HISTOGRAM_SIZE 256

namespace cg = cooperative_groups;

extern "C" void equalizeHistCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height);
extern "C" void equalizeHistCuda(const uint8_t *src, uint8_t *dst, int channels, int width, int height);

__global__ void calculateHist(const uint8_t *src, int *hist, int channels, int width, int height);
__global__ void calculateHistSum(const int *hist, int *histSum, int channels);
__global__ void equalizeHist(const uint8_t *src, uint8_t *dst, const int *histSum, int channels, int width, int height);

__global__ void calculateHist(const uint8_t *src, int *hist, int channels, int width, int height) {	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	cg::thread_block g = cg::this_thread_block();

	if (x >= width * height)
		return;

	extern __shared__ unsigned int histShared[]; // HISTOGRAM_SIZE * channels

	if(threadIdx.x < HISTOGRAM_SIZE) {
		for (int c = 0; c < channels; c++) {
			histShared[threadIdx.x * channels + c] = 0;
		}
	}

	g.sync();

	for (int c = 0; c < channels; c++) {
		atomicAdd(&histShared[int(src[x * channels + c]) * channels + c], 1);
	}

	g.sync();

	if(threadIdx.x < HISTOGRAM_SIZE) {
		for (int c = 0; c < channels; c++) {
			atomicAdd(&hist[threadIdx.x * channels + c], histShared[threadIdx.x * channels + c]);
		}
	}
}

__global__ void calculateHistSum(const int *hist, int *histSum, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	cg::thread_block g = cg::this_thread_block();

	extern __shared__ unsigned int histShared[]; // HISTOGRAM_SIZE * channels

	for (int c = 0; c < channels; c++) {
		histShared[x * channels + c] = hist[x * channels + c];
	}

	g.sync();

	// Kogge-Stone Scan
	for (int stride = 1; stride < HISTOGRAM_SIZE; stride *= 2) {
		g.sync();
		if (threadIdx.x >= stride) {
			for (int c = 0; c < channels; c++) {
				histShared[threadIdx.x * channels + c] += histShared[(threadIdx.x - stride) * channels + c];
			}
		}
	}

	g.sync();

	for (int c = 0; c < channels; c++) {
		histSum[x * channels + c] = histShared[x * channels + c];
	}
}

__global__ void equalizeHist(const uint8_t *src, uint8_t *dst, const int *histSum, int channels, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	cg::thread_block g = cg::this_thread_block();

	if (x >= width * height)
		return;

	extern __shared__ unsigned int histShared[]; // HISTOGRAM_SIZE * channels

	if (threadIdx.x < HISTOGRAM_SIZE) {
		for (int c = 0; c < channels; c++) {
			histShared[threadIdx.x * channels + c] = histSum[threadIdx.x * channels + c];
		}
	}

	g.sync();

	for (int c = 0; c < channels; c++) {
		dst[x * channels + c] = (uint8_t)lroundf(histShared[src[x * channels + c] * channels + c] * 255 / float(width * height));
	}
}

void equalizeHistCPU(const uint8_t *src, uint8_t *dst, int channels, int width, int height) {
	std::vector<int> hist(HISTOGRAM_SIZE * channels, 0);
	std::vector<int> histSum(HISTOGRAM_SIZE * channels, 0);

	// calculate histogram
	for (int y = 0; y < height; y++) {
		const uint8_t *srcRow = src + y * width * channels;
		for (int x = 0; x < width; x++) {
			for (int c = 0; c < channels; c++) {
				hist[srcRow[x * channels + c] * channels + c]++;
			}
		}
	}

	// calculate cumulative histogram
	histSum[0] = hist[0];
	for (int y = 1; y < HISTOGRAM_SIZE; y++) {
		for (int c = 0; c < channels; c++) {
			histSum[y * channels + c] = hist[y * channels + c] + histSum[(y - 1) * channels + c];
		}
	}

	// equalize histogram
	for (int y = 0; y < height; y++) {
		const uint8_t *srcRow = src + y * width * channels;
		uint8_t *distRow = dst + y * width * channels;
		for (int x = 0; x < width; x++) {
			for (int c = 0; c < channels; c++) {
				distRow[x * channels + c] = (uint8_t)std::round(histSum[srcRow[x * channels + c] * channels + c] * 255 / float(width * height));
			}
		}
	}
}

void equalizeHistCuda(const uint8_t *devSrc, uint8_t *devDst, int channels, int width, int height) {
	// Reference:
	//     https://github.com/nuwandda/cuda-histogram-equalization/blob/main/kernel.cu

	int *devHist, *devHistSum{};
	int size = HISTOGRAM_SIZE * channels * sizeof(int);

	checkCudaErrors(cudaMalloc(&devHist, size));
	checkCudaErrors(cudaMalloc(&devHistSum, size));

	dim3 threadsPerBlock(HISTOGRAM_SIZE * 4);
	dim3 numBlocks(divUp(width * height, threadsPerBlock.x));
	calculateHist<<<numBlocks, threadsPerBlock, size>>>(devSrc, devHist, channels, width, height);

	calculateHistSum<<<1, HISTOGRAM_SIZE, size>>>(devHist, devHistSum, channels);

	equalizeHist<<<numBlocks, threadsPerBlock, size>>>(devSrc, devDst, devHistSum, channels, width, height);

	cudaFree(devHist);
	cudaFree(devHistSum);
}