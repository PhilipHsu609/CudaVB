#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "my_cuda_lib.h"
#include "helper_cuda.h"
#include "utils.h"

#include <cstdint>

using morphOp = uint8_t(*)(uint8_t, uint8_t);

extern "C" void dilateCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize);
extern "C" void dilateCuda(const uint8_t *devSrc, uint8_t *devDst, int channels, int width, int height, const int *kernel, int kernelSize);

extern "C" void erodeCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize);
extern "C" void erodeCuda(const uint8_t *devSrc, uint8_t *devDst, int channels, int width, int height, const int *kernel, int kernelSize);

void morphCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize, morphOp op);
void morphCuda(const uint8_t *devSrc, uint8_t *devDst, int channels, int width, int height, const int *kernel, int kernelSize, morphOp op);

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
    int kernelRadius = kernelSize >> 1;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height)
        return;

    for(int c = 0; c < channels; c++) {
        uint8_t p = src[(y * width + x) * channels + c];

        for(int ky = 0; ky < kernelSize; ky++) {
            for(int kx = 0; kx < kernelSize; kx++) {
                int srcX = x + kx - kernelRadius;
                int srcY = y + ky - kernelRadius;

                if(srcX < 0 || srcX >= width || srcY < 0 || srcY >= height)
                    continue;

                if(kernel[ky * kernelSize + kx])
                    p = op(p, src[(srcY * width + srcX) * channels + c]);
            }
        }

        dst[(y * width + x) * channels + c] = p;
    }
}

void dilateCuda(const uint8_t *devSrc, uint8_t *devDst, int channels, int width, int height, const int *kernel, int kernelSize) {
    morphOp h_dilateOp;
    checkCudaErrors(cudaMemcpyFromSymbol(&h_dilateOp, d_dilateOp, sizeof(morphOp)));
    morphCuda(devSrc, devDst, channels, width, height, kernel, kernelSize, h_dilateOp);
}

void erodeCuda(const uint8_t *devSrc, uint8_t *devDst, int channels, int width, int height, const int *kernel, int kernelSize) {
    morphOp h_erodeOp;
    checkCudaErrors(cudaMemcpyFromSymbol(&h_erodeOp, d_erodeOp, sizeof(morphOp)));
    morphCuda(devSrc, devDst, channels, width, height, kernel, kernelSize, h_erodeOp);
}

void dilateCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize) {
    morphCpu(src, dst, channels, width, height, kernel, kernelSize, dilateOp);
}

void erodeCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize) {
    morphCpu(src, dst, channels, width, height, kernel, kernelSize, erodeOp);
}

void morphCuda(const uint8_t *devSrc, uint8_t *devDst, int channels, int width, int height, const int *kernel, int kernelSize, morphOp op) {
    int *devKernel;

    checkCudaErrors(cudaMalloc(&devKernel, kernelSize * kernelSize * sizeof(int)));
    checkCudaErrors(cudaMemcpy(devKernel, kernel, kernelSize * kernelSize * sizeof(int), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(divUp(width, threadsPerBlock.x), divUp(height, threadsPerBlock.y));
    morphKernel<<<numBlocks, threadsPerBlock>>>(devSrc, devDst, channels, width, height, devKernel, kernelSize, op);

    checkCudaErrors(cudaFree(devKernel));
}

void morphCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const int *kernel, int kernelSize, morphOp op) {
    int kernelRadius = kernelSize >> 1;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			for (int c = 0; c < channels; c++) {
				uint8_t p = src[(y * width + x) * channels + c];

				for (int ky = 0; ky < kernelSize; ky++) {
					for (int kx = 0; kx < kernelSize; kx++) {
						int srcX = x + kx - kernelRadius;
						int srcY = y + ky - kernelRadius;

                        if(srcX < 0 || srcX >= width || srcY < 0 || srcY >= height)
                            continue;

						if(kernel[ky * kernelSize + kx])
							p = op(p, src[(srcY * width + srcX) * channels + c]);
					}
				}

				dst[(y * width + x) * channels + c] = p;
			}
		}
	}
}