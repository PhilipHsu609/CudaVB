#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "my_cuda_lib.h"
#include "helper_cuda.h"
#include "utils.h"

#include <iostream>
#include <cstdint>
#include <cmath>
#include <vector>

extern "C" void conv2DCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const float *kernel, int kernelSize);
extern "C" void conv2DCuda(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const float *kernel, int kernelSize);

__global__ void conv2DKernel(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const float *kernel, int kernelSize);

__global__ void conv2DKernel(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const float *kernel, int kernelSize) {
	int kernelRadius = kernelSize >> 1;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (y >= height || x >= width)
		return;

	for (int c = 0; c < channels; c++) {
		float sum = 0.0f;

		for (int ky = 0; ky < kernelSize; ky++) {
			for (int kx = 0; kx < kernelSize; kx++) {
				int srcX = x + kx - kernelRadius;
				int srcY = y + ky - kernelRadius;

				if(srcX < 0 || srcX >= width || srcY < 0 || srcY >= height)
                    continue;

				sum += src[(srcY * width + srcX) * channels + c] * kernel[ky * kernelSize + kx];
			}
		}

		dst[(y * width + x) * channels + c] = (uint8_t)lroundf(sum);
	}
}

void conv2DCuda(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const float *kernel, int kernelSize) {
	float *d_kernel;
	uint8_t *d_src, *d_dst;

	checkCudaErrors(cudaMalloc(&d_kernel, sizeof(float) * kernelSize * kernelSize));
	checkCudaErrors(cudaMalloc(&d_dst, sizeof(uint8_t) * width * height * channels));
	checkCudaErrors(cudaMalloc(&d_src, sizeof(uint8_t) * width * height * channels));

	checkCudaErrors(cudaMemcpy(d_kernel, kernel, sizeof(float) * kernelSize * kernelSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_src, src, sizeof(uint8_t) * width * height * channels, cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(divUp(width, threadsPerBlock.x), divUp(height, threadsPerBlock.y));
	conv2DKernel<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, channels, width, height, d_kernel, kernelSize);

	checkCudaErrors(cudaMemcpy(dst, d_dst, sizeof(uint8_t) * width * height * channels, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_kernel));
	checkCudaErrors(cudaFree(d_src));
	checkCudaErrors(cudaFree(d_dst));
}

void conv2DCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const float *kernel, int kernelSize) {
	int kernelRadius = kernelSize >> 1;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			for (int c = 0; c < channels; c++) {
				float sum = 0.0f;

				for (int ky = 0; ky < kernelSize; ky++) {
					for (int kx = 0; kx < kernelSize; kx++) {
						int srcX = x + kx - kernelRadius;
						int srcY = y + ky - kernelRadius;

						if(srcX < 0 || srcX >= width || srcY < 0 || srcY >= height)
                            continue;

						sum += src[(srcY * width + srcX) * channels + c] * kernel[ky * kernelSize + kx];
					}
				}

				dst[(y * width + x) * channels + c] = (uint8_t)std::round(sum);
			}
		}
	}
}