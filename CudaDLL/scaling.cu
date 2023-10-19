#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "my_cuda_lib.h"
#include "helper_cuda.h"
#include "utils.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>

extern "C" void pyrUpCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height);
//extern "C" void pyrUpCuda(const uint8_t *src, uint8_t *dst, int channels, int width, int height);

extern "C" void pyrDownCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height);
//extern "C" void pyrDownCuda(const uint8_t *src, uint8_t *dst, int channels, int width, int height);

extern "C" void bilinearCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, int dstWidth, int dstHeight);
extern "C" void bilinearCuda(const uint8_t * src, uint8_t * dst, int channels, int width, int height, int dstWidth, int dstHeight);

__global__ void bilinearKernel(cudaTextureObject_t texObj, uint8_t *dst, int channels, int width, int height, int dstWidth, int dstHeight);
__global__ void bilinearKernel(const uint8_t *src, uint8_t *dst, int channels, int width, int height, int dstWidth, int dstHeight);

__global__ void bilinearKernel(cudaTextureObject_t texObj, uint8_t *dst, int channels, int width, int height, int dstWidth, int dstHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= dstWidth || y >= dstHeight)
		return;

	float scaleX = float(width) / float(dstWidth);
	float scaleY = float(height) / float(dstHeight);

	float fx = (x - 0.5f) * scaleX + 0.5f;
	float fy = (y - 0.5f) * scaleY + 0.5f;

	for(int c = 0; c < channels; c++)
		dst[(y * dstWidth + x) * channels + c] = tex2D<uint8_t>(texObj, channels * fx + c, fy);
}

__global__ void bilinearKernel(const uint8_t *src, uint8_t *dst, int channels, int width, int height, int dstWidth, int dstHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= dstWidth || y >= dstHeight)
		return;

	float fx = float(x) / float(dstWidth) * float(width) - 0.5f;
	float fy = float(y) / float(dstHeight) * float(height) - 0.5f;

	fx = fx < 0.0f ? 0.0f : fx;
	fy = fy < 0.0f ? 0.0f : fy;

	int x1 = int(fx);
	int y1 = int(fy);

	int x2 = x1 >= width - 1 ? width - 1 : x1 + 1;
	int y2 = y1 >= height - 1 ? height - 1 : y1 + 1;

	fx -= float(x1);
	fy -= float(y1);

	for(int c = 0; c < channels; c++) {
		int s00 = src[(y1 * width + x1) * channels + c];
		int s01 = src[(y1 * width + x2) * channels + c];
		int s10 = src[(y2 * width + x1) * channels + c];
		int s11 = src[(y2 * width + x2) * channels + c];

		int t0 = (1.0f - fx) * s00 + fx * s01;
		int t1 = (1.0f - fx) * s10 + fx * s11;
		int t = (1.0f - fy) * t0 + fy * t1;

		dst[(y * dstWidth + x) * channels + c] = uint8_t(t);
	}
}

void pyrUpCuda(const uint8_t *src, uint8_t *dst, int channels, int width, int height) {
	//throw std::runtime_error("Not implemented");
}

void pyrDownCuda(const uint8_t *src, uint8_t *dst, int channels, int width, int height) {
	//throw std::runtime_error("Not implemented");
}

void pyrUpCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height) {
	// Reference:
	//     https://github.com/opencv/opencv/blob/7b6d65cf201106de5cd9b3bebdcc939da115b654/modules/imgproc/src/pyramids.cpp#L1042

	const int PU_SZ = 3;

	int dstWidth = width * 2;
	int cn = channels;

	int sy0 = -PU_SZ / 2, sy = sy0;
	int *buf = new int[dstWidth * cn * PU_SZ];
	int *dtab = new int[width * cn];

	for (int x = 0; x < width * cn; x++) {
		dtab[x] = (x / cn) * 2 * cn + x % cn;
	}

	int tmp0, tmp1;
	int *rows[PU_SZ]; // for vertical convolution
	for (int y = 0; y < height; y++) {
		uint8_t *dst0 = dst + y * 2 * dstWidth * cn;        // dst even row
		uint8_t *dst1 = dst + (y * 2 + 1) * dstWidth * cn;  // dst odd row

		// horizontal convolution and decimation
		for (; sy <= y + 1; sy++) {
			int *row = buf + ((sy - sy0) % PU_SZ) * dstWidth * cn;    // ((sy - sy0) % PU_SZ): 0, 1, 2 repeat
			int _sy = borderInterpolate(sy * 2, height * 2, 3) / 2;
			const uint8_t *_src = src + _sy * width * cn;

			// left, right borders
			for (int x = 0; x < cn; x++) {
				int dx = dtab[x];

				tmp0 = _src[x] * 6 + _src[x + cn] * 2;   // left most
				tmp1 = (_src[x] + src[x + cn]) * 4;      // second from the left

				row[dx] = tmp0, row[dx + cn] = tmp1;

				int sx = width * cn - cn + x;
				dx = dtab[sx];

				tmp0 = _src[sx - cn] + _src[sx] * 7;    // second from the right
				tmp1 = _src[sx] * 8;                    // right most

				row[dx] = tmp0, row[dx + cn] = tmp1;
			}

			// center part
			for (int x = cn; x < width * cn - cn; x++) {
				int dx = dtab[x];
				tmp0 = _src[x - cn] + _src[x] * 6 + _src[x + cn];
				tmp1 = (_src[x] + _src[x + cn]) * 4;

				row[dx] = tmp0, row[dx + cn] = tmp1;
			}
		}

		// vertical convolution and decimation
		for (int k = 0; k < PU_SZ; k++) {
			rows[k] = buf + ((y - PU_SZ / 2 + k - sy0) % PU_SZ) * dstWidth * cn; // ((y - PU_SZ / 2 + k - sy0) % PU_SZ): 0, 1, 2 -> 1, 2, 0 -> 2, 0, 1 -> 0, 1, 2
		}

		for (int x = 0; x < dstWidth * cn; x++) {
			tmp0 = rows[0][x] + rows[1][x] * 6 + rows[2][x];
			tmp1 = (rows[1][x] + rows[2][x]) * 4;

			tmp0 = (tmp0 + (1 << 5)) >> 6;  // rounding, divided by 64
			tmp1 = (tmp1 + (1 << 5)) >> 6;

			dst0[x] = (uint8_t)tmp0, dst1[x] = (uint8_t)tmp1;
		}
	}

	delete[] buf, dtab;
}

void pyrDownCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height) {
	// Reference:
	//     https://github.com/opencv/opencv/blob/7b6d65cf201106de5cd9b3bebdcc939da115b654/modules/imgproc/src/pyramids.cpp#L884

	const int PD_SZ = 5;

	int dstWidth = width / 2;
	int dstHeight = height / 2;
	int cn = channels;

	int *buf = new int[dstWidth * cn * PD_SZ];
	int *tabL = new int[cn * (PD_SZ + 2)];
	int *tabR = new int[cn * (PD_SZ + 2)];

	int sy0 = -PD_SZ / 2, sy = sy0;

	for (int x = 0; x <= PD_SZ + 1; x++) {
		int sx0 = borderInterpolate(x - PD_SZ / 2, width, 3) * cn;
		int sx1 = borderInterpolate(x + width * 2 - PD_SZ / 2, width, 3) * cn;

		for (int k = 0; k < cn; k++) {
			tabL[x * cn + k] = sx0 + k;
			tabR[x * cn + k] = sx1 + k;
		}
	}

	int *rows[PD_SZ];
	for (int y = 0; y < dstHeight; y++) {
		uint8_t *_dst = dst + y * dstWidth * cn;

		for (; sy <= y * 2 + 2; sy++) {
			int *row = buf + ((sy - sy0) % PD_SZ) * dstWidth * cn;
			int _sy = borderInterpolate(sy, height, 3);
			const uint8_t *_src = src + _sy * width * cn;

			for (int x = 0; x < cn; x++) {
				row[x] = _src[tabL[x + cn * 2]] * 6 + (_src[tabL[x + cn]] + _src[tabL[x + cn * 3]]) * 4 + _src[tabL[x]] + _src[tabL[x + cn * 4]];
			}

			if (cn == 1) {
				for (int x = cn; x < dstWidth * cn; x++) {
					row[x] = _src[x * 2] * 6 + (_src[x * 2 - 1] + _src[x * 2 + 1]) * 4 + _src[x * 2 - 2] + _src[x * 2 + 2];
				}
			} else if (cn == 3) {
				for (int x = cn; x < dstWidth * cn; x += cn) {
					const uint8_t *s = _src + x * 2;
					int t0 = s[0] * 6 + (s[-3] + s[3]) * 4 + s[-6] + s[6];
					int t1 = s[1] * 6 + (s[-2] + s[4]) * 4 + s[-5] + s[7];
					int t2 = s[2] * 6 + (s[-1] + s[5]) * 4 + s[-4] + s[8];
					row[x] = t0, row[x + 1] = t1, row[x + 2] = t2;
				}
			} else {
				//throw std::runtime_error("Unsupported number of channels");
			}
		}

		for (int k = 0; k < PD_SZ; k++) {
			rows[k] = buf + ((y * 2 - PD_SZ / 2 + k - sy0) % PD_SZ) * dstWidth * cn;
		}

		for (int x = 0; x < dstWidth * cn; x++) {
			int t = rows[2][x] * 6 + (rows[1][x] + rows[3][x]) * 4 + rows[0][x] + rows[4][x];
			t = (t + (1 << 7)) >> 8;
			_dst[x] = (uint8_t)t;
		}
	}

	delete[] buf, tabL, tabR;
}

void bilinearCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, int dstWidth, int dstHeight) {
	float scaleX = (float)width / dstWidth;
	float scaleY = (float)height / dstHeight;

	float fx, fy;
	int sx, sy;
	for (int dx = 0; dx < dstWidth; dx++) {
		fx = (dx + 0.5f) * scaleX - 0.5f;
		sx = (int)fx;
		fx -= sx;

		if(sx < 0)
			fx = 0, sx = 0;

		if(sx >= width - 1)
			fx = 0, sx = width - 1;

		short cbufx[2];
		cbufx[0] = (short)((1.f - fx) * 2048);
		cbufx[1] = 2048 - cbufx[0];

		for (int dy = 0; dy < dstHeight; dy++) {
			fy = (dy + 0.5f) * scaleY - 0.5f;
			sy = (int)fy;
			fy -= sy;

			short cbufy[2];
			cbufy[0] = (short)((1.f - fy) * 2048);
			cbufy[1] = 2048 - cbufy[0];

			for (int c = 0; c < channels; c++) {
				int s00 = src[(sy * width + sx) * channels + c];
				int s01 = src[(sy * width + sx + 1) * channels + c];
				int s10 = src[((sy + 1) * width + sx) * channels + c];
				int s11 = src[((sy + 1) * width + sx + 1) * channels + c];

				int t0 = cbufx[0] * s00 + cbufx[1] * s01;
				int t1 = cbufx[0] * s10 + cbufx[1] * s11;
				int t = (cbufy[0] * t0 + cbufy[1] * t1) >> 22;

				dst[(dy * dstWidth + dx) * channels + c] = (uint8_t)t;
			}
		}
	}
}

void bilinearCuda(const uint8_t *src, uint8_t *dst, int channels, int width, int height, int dstWidth, int dstHeight) {
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(divUp(dstWidth, threadsPerBlock.x), divUp(dstHeight, threadsPerBlock.y));
	bilinearKernel<<<numBlocks, threadsPerBlock>>>(src, dst, channels, width, height, dstWidth, dstHeight);
}

void bilinearCudaTexture(const uint8_t *src, uint8_t *dst, int channels, int width, int height, int dstWidth, int dstHeight) {
	// TODO: Needs to convert src to uchar4 first

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaArray_t cuArray;
	checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, width * channels, height));

	const size_t spitch = width * channels * sizeof(uint8_t);
	checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, src, spitch, width * channels * sizeof(uint8_t), height, cudaMemcpyHostToDevice));

	cudaResourceDesc resDesc;
	std::memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaTextureDesc texDesc;
	std::memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 0;

	cudaTextureObject_t texObj = 0;
	checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

	uint8_t *devDst;
	checkCudaErrors(cudaMalloc(&devDst, dstWidth * dstHeight * channels * sizeof(uint8_t)));

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(divUp(dstWidth, threadsPerBlock.x), divUp(dstHeight, threadsPerBlock.y));
	bilinearKernel<<<numBlocks, threadsPerBlock>>>(texObj, devDst, channels, width, height, dstWidth, dstHeight);

	checkCudaErrors(cudaMemcpy(dst, devDst, dstWidth * dstHeight * channels * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaDestroyTextureObject(texObj));
	checkCudaErrors(cudaFree(devDst));
	checkCudaErrors(cudaFreeArray(cuArray));
}