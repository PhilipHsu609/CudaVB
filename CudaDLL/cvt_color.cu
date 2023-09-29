#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "my_cuda_lib.h"
#include "helper_cuda.h"

#include <iostream>
#include <cstdint>
#include <cmath>

extern "C" void bgr2hsvCpu(const uint8_t *src, uint8_t *dst, const size_t width, const size_t height);
extern "C" void bgr2grayCpu(const uint8_t * src, uint8_t * dst, const size_t width, const size_t height);

extern "C" void bgr2hsvCuda(const uint8_t * src, uint8_t * dst, const size_t width, const size_t height);
extern "C" void bgr2grayCuda(const uint8_t * src, uint8_t * dst, const size_t width, const size_t height);

__global__ void bgr2hsvKernel(const uint8_t *src, uint8_t *dst, const size_t width, const size_t height);
__global__ void bgr2grayKernel(const uint8_t *src, uint8_t *dst, const size_t width, const size_t height);

__global__ void bgr2hsvKernel(const uint8_t *src, uint8_t *dst, const size_t width, const size_t height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int i = y * width + x;
		float h, s, v;
		float b = src[3 * i] / 255.0f;
		float g = src[3 * i + 1] / 255.0f;
		float r = src[3 * i + 2] / 255.0f;

		float max = r > g ? (r > b ? r : b) : (g > b ? g : b);
		float min = r < g ? (r < b ? r : b) : (g < b ? g : b);

		v = max;
		float delta = v - min;

		s = (v == 0) ? 0 : delta / v;

		if (v == r) {
			h = 60 * (g - b) / delta;
		} else if (v == g) {
			h = 120 + 60 * (b - r) / delta;
		} else if (v == b) {
			h = 240 + 60 * (r - g) / delta;
		} else {
			h = 0;
		}

		if (h < 0) {
			h += 360;
		}

		dst[3 * i] = (uint8_t)lroundf(h / 2.0f);
		dst[3 * i + 1] = (uint8_t)lroundf(s * 255.0f);
		dst[3 * i + 2] = (uint8_t)lroundf(v * 255.0f);
	}
}

__global__ void bgr2grayKernel(const uint8_t *src, uint8_t *dst, const size_t width, const size_t height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int i = y * width + x;
		dst[i] = (uint8_t)lroundf(0.114f * src[3 * i] + 0.587f * src[3 * i + 1] + 0.299f * src[3 * i + 2]);
	}
}

void bgr2hsvCuda(const uint8_t *src, uint8_t *dst, const size_t width, const size_t height) {
	uint8_t *d_src, *d_dst;

	checkCudaErrors(cudaMalloc(&d_src, width * height * 3 * sizeof(uint8_t)));
	checkCudaErrors(cudaMalloc(&d_dst, width * height * 3 * sizeof(uint8_t)));

	checkCudaErrors(cudaMemcpy(d_src, src, width * height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	bgr2hsvKernel<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, width, height);

	checkCudaErrors(cudaMemcpy(dst, d_dst, width * height * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_src));
	checkCudaErrors(cudaFree(d_dst));
}

void bgr2grayCuda(const uint8_t *src, uint8_t *dst, const size_t width, const size_t height) {
	uint8_t *d_src, *d_dst;

	checkCudaErrors(cudaMalloc(&d_src, width * height * 3 * sizeof(uint8_t)));
	checkCudaErrors(cudaMalloc(&d_dst, width * height * sizeof(uint8_t)));

	checkCudaErrors(cudaMemcpy(d_src, src, width * height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	bgr2grayKernel<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, width, height);

	checkCudaErrors(cudaMemcpy(dst, d_dst, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_src));
	checkCudaErrors(cudaFree(d_dst));
}

void bgr2hsvCpu(const uint8_t *src, uint8_t *dst, const size_t width, const size_t height) {
	for (size_t i = 0; i < width * height; i++) {
		float h, s, v;
		float b = src[3 * i] / 255.0f;
		float g = src[3 * i + 1] / 255.0f;
		float r = src[3 * i + 2] / 255.0f;

		float max = r > g ? (r > b ? r : b) : (g > b ? g : b);
		float min = r < g ? (r < b ? r : b) : (g < b ? g : b);

		v = max;
		float delta = v - min;

		s = (v == 0) ? 0 : delta / v;

		if (v == r) {
			h = 60 * (g - b) / delta;
		} else if (v == g) {
			h = 120 + 60 * (b - r) / delta;
		} else if (v == b) {
			h = 240 + 60 * (r - g) / delta;
		} else {
			h = 0;
		}

		if (h < 0) {
			h += 360;
		}

		dst[3 * i] = (uint8_t)std::round(h / 2.0f);
		dst[3 * i + 1] = (uint8_t)std::round(s * 255.0f);
		dst[3 * i + 2] = (uint8_t)std::round(v * 255.0f);
	}
}

void bgr2grayCpu(const uint8_t *src, uint8_t *dst, const size_t width, const size_t height) {
	for (size_t i = 0; i < width * height; i++) {
		dst[i] = (uint8_t)std::round(0.114f * src[3 * i] + 0.587f * src[3 * i + 1] + 0.299f * src[3 * i + 2]);
	}
}