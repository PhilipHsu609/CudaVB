#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "my_cuda_lib.h"
#include "helper_cuda.h"
#include "utils.h"

#include <cstdint>

using morphOp = uint8_t(*)(uint8_t, uint8_t);

extern "C" void dilateCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize);
extern "C" void dilateCuda(const uint8_t * src, uint8_t * dst, int channels, int width, int height, const int *kernel, int kernelSize);

extern "C" void erodeCpu(const uint8_t * src, uint8_t * dst, int channels, int width, int height, const int *kernel, int kernelSize);
extern "C" void erodeCuda(const uint8_t * src, uint8_t * dst, int channels, int width, int height, const int *kernel, int kernelSize);

void morphCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize, morphOp op);
void morphCuda(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize, morphOp op);

__host__ __device__ uint8_t dilateOp(uint8_t p1, uint8_t p2);
__host__ __device__ uint8_t erodeOp(uint8_t p1, uint8_t p2);

__device__ morphOp d_dilateOp = dilateOp;
__device__ morphOp d_erodeOp = erodeOp;

__global__ void morphKernel(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize, morphOp op);

__host__ __device__ uint8_t dilateOp(uint8_t p1, uint8_t p2) {
	// max(p1, p2)
    return p1 >= p2 ? p1 : p2;
}
__host__ __device__ uint8_t erodeOp(uint8_t p1, uint8_t p2) {
	// min(p1, p2)
    return p1 <= p2 ? p1 : p2;
}

__global__ void morphKernel(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize, morphOp op) {
    int kernelRadius = kernelSize / 2;
    int paddedWidth = width + 2 * kernelRadius;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height)
        return;

    for(int c = 0; c < channels; c++) {
        uint8_t p = src[((y + kernelRadius) * paddedWidth + x + kernelRadius) * channels + c];

        for(int ky = 0; ky < kernelSize; ky++) {
            for(int kx = 0; kx < kernelSize; kx++) {
                int srcX = x + kx;
                int srcY = y + ky;

                if(kernel[ky * kernelSize + kx])
                    p = op(p, src[(srcY * paddedWidth + srcX) * channels + c]);
            }
        }

        dst[(y * width + x) * channels + c] = p;
    }
}

void dilateCuda(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize) {
    morphOp h_dilateOp;
    checkCudaErrors(cudaMemcpyFromSymbol(&h_dilateOp, d_dilateOp, sizeof(morphOp)));
    morphCuda(src, dst, channels, width, height, kernel, kernelSize, h_dilateOp);
}

void erodeCuda(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize) {
    morphOp h_erodeOp;
    checkCudaErrors(cudaMemcpyFromSymbol(&h_erodeOp, d_erodeOp, sizeof(morphOp)));
    morphCuda(src, dst, channels, width, height, kernel, kernelSize, h_erodeOp);
}

void dilateCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize) {
    morphCpu(src, dst, channels, width, height, kernel, kernelSize, dilateOp);
}

void erodeCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize) {
    morphCpu(src, dst, channels, width, height, kernel, kernelSize, erodeOp);
}

void morphCuda(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize, morphOp op) {
    int *d_kernel;
    uint8_t *d_pad, *d_dst;
    int kernelRadius = kernelSize >> 1;

    auto paddedSrc{ padding2D(src, channels, width, height, kernelRadius, kernelRadius, kernelRadius, kernelRadius, 0, 0xff) };

    checkCudaErrors(cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_dst, width * height * channels * sizeof(uint8_t)));
    checkCudaErrors(cudaMalloc(&d_pad, paddedSrc.size() * sizeof(uint8_t)));

    checkCudaErrors(cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_pad, paddedSrc.data(), paddedSrc.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(divUp(width, threadsPerBlock.x), divUp(height, threadsPerBlock.y));
    morphKernel<<<numBlocks, threadsPerBlock>>>(d_pad, d_dst, channels, width, height, d_kernel, kernelSize, op);

    checkCudaErrors(cudaMemcpy(dst, d_dst, width * height * channels * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_kernel));
    checkCudaErrors(cudaFree(d_dst));
    checkCudaErrors(cudaFree(d_pad));
}

void morphCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize, morphOp op) {
    int kernelRadius = kernelSize / 2;

    auto paddedSrc{ padding2D(src, channels, width, height, kernelRadius, kernelRadius, kernelRadius, kernelRadius, 0, 0xff) };

	int paddedWidth = width + 2 * kernelRadius;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			for (int c = 0; c < channels; c++) {
				uint8_t p = paddedSrc[((y + kernelRadius) * paddedWidth + x + kernelRadius) * channels + c];

				for (int ky = 0; ky < kernelSize; ky++) {
					for (int kx = 0; kx < kernelSize; kx++) {
						int srcX = x + kx;
						int srcY = y + ky;

						if(kernel[ky * kernelSize + kx])
							p = op(p, paddedSrc[(srcY * paddedWidth + srcX) * channels + c]);
					}
				}

				dst[(y * width + x) * channels + c] = p;
			}
		}
	}
}