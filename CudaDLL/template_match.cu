#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "my_cuda_lib.h"
#include "cuda_runtime.h"
#include "npp.h"
#include "helper_cuda.h"
#include "utils.h"

#include <cmath>

extern "C" void matchTemplateCpu(const uint8_t *src, float *dst, int channels, int width, int height, const uint8_t *templ, int tWidth, int tHeight);
extern "C" void matchTemplateCuda(const uint8_t *src, float *dst, int channels, int width, int height, const uint8_t *templ, int tWidth, int tHeight);

__global__ void sumChannels(const float *src, float *dst, int width, int height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float sum = 0.0f;
		for (int c = 0; c < channels; c++) {
			sum += src[(y * width + x) * channels + c];
		}
		dst[y * width + x] = sum;
	}
}

void matchTemplateCuda(const uint8_t *src, float *dst, int channels, int width, int height, const uint8_t *templ, int tWidth, int tHeight) {
	NppiSize oSrcSize{ width, height };
	NppiSize oSrcRoiSize{ width, height };

	NppiSize oTplRoiSize{ tWidth, tHeight };

	int nBufferSize;
	Npp8u *pBuffer{};

	if (channels == 1) {
		nppiValidNormLevelGetBufferHostSize_8u32f_C1R(oSrcRoiSize, &nBufferSize);
		cudaMalloc(&pBuffer, nBufferSize);
		nppiCrossCorrValid_NormLevel_8u32f_C1R(
			src, width * sizeof(uint8_t), oSrcRoiSize,
			templ, tWidth * sizeof(uint8_t), oTplRoiSize,
			dst, (width - tWidth + 1) * sizeof(float), pBuffer
		);
	} else if (channels == 3) {
		float *dstC3{};
		cudaMalloc(&dstC3, (width - tWidth + 1) * (height - tHeight + 1) * channels * sizeof(float));

		nppiValidNormLevelGetBufferHostSize_8u32f_C3R(oSrcRoiSize, &nBufferSize);
		cudaMalloc(&pBuffer, nBufferSize);
		checkNPPErrors(nppiCrossCorrValid_NormLevel_8u32f_C3R(
			src, width * channels * sizeof(uint8_t), oSrcRoiSize,
			templ, tWidth * channels * sizeof(uint8_t), oTplRoiSize,
			dstC3, (width - tWidth + 1) * channels * sizeof(float), pBuffer
		));

		// sum channels
		dim3 threadsPerBlock(16, 16);
		dim3 numBlocks(divUp(width - tWidth + 1, threadsPerBlock.x),
						divUp(height - tHeight + 1, threadsPerBlock.y));

		sumChannels<<<numBlocks, threadsPerBlock>>>(dstC3, dst, width - tWidth + 1, height - tHeight + 1, channels);
	}

	cudaFree(pBuffer);
}

void integral(const uint8_t *src, uint32_t *dst, int width, int height) {
	for (int y = 1; y <= height; y++) {
		uint32_t sum = 0;
		for (int x = 1; x <= width; x++) {
			sum += src[(y - 1) * width + x - 1];
			dst[y * (width + 1) + x] = sum + dst[(y - 1) * (width + 1) + x];
		}
	}
}

void sqrIntegral(const uint8_t *src, uint64_t *dst, int width, int height) {
	for (int y = 1; y <= height; y++) {
		uint64_t sum = 0;
		for (int x = 1; x <= width; x++) {
			uint64_t val = src[(y - 1) * width + x - 1];
			sum += val * val;
			dst[y * (width + 1) + x] = sum + dst[(y - 1) * (width + 1) + x];
		}
	}
}

std::vector<uint32_t> sum(const uint8_t *src, int channels, int width, int height) {
	std::vector<uint32_t> res(channels, 0);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int c = 0; c < channels; c++) {
				res[c] += src[(y * width + x) * channels + c];
			}
		}
	}
	return res;
}

std::vector<uint64_t> sqrSum(const uint8_t *src, int channels, int width, int height) {
	std::vector<uint64_t> res(channels, 0);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int c = 0; c < channels; c++) {
				uint64_t val = src[(y * width + x) * channels + c];
				res[c] += val * val;
			}
		}
	}
	return res;
}

void matchCCORR(const uint8_t *src, float *dst, int channels, int width, int height, const uint8_t *templ, int tWidth, int tHeight) {
	for (int y = 0; y < height - tHeight + 1; y++) {
		for (int x = 0; x < width - tWidth + 1; x++) {
			float ccorr = 0.0f;
			for (int ty = 0; ty < tHeight; ty++) {
				for (int tx = 0; tx < tWidth; tx++) {
					for (int c = 0; c < channels; c++) {
						ccorr += src[((y + ty) * width + x + tx) * channels + c] * templ[(ty * tWidth + tx) * channels + c];
					}
				}
			}
			dst[y * (width - tWidth + 1) + x] = ccorr;
		}
	}
}

void matchTemplateCpu(const uint8_t *src, float *dst, int channels, int width, int height, const uint8_t *templ, int tWidth, int tHeight) {
	std::vector<std::vector<uint8_t>> images_(channels, std::vector<uint8_t>(width * height));
	std::vector<std::vector<uint32_t>> imageSums_(channels, std::vector<uint32_t>((width + 1) * (height + 1)));
	std::vector<std::vector<uint64_t>> imageSqrSums_(channels, std::vector<uint64_t>((width + 1) * (height + 1)));

	// 1. split image into channels
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int c = 0; c < channels; c++) {
				images_[c][y * width + x] = src[(y * width + x) * channels + c];
			}
		}
	}

	// 2. calculate integral images for each channel
	for (int i = 0; i < channels; i++) {
		integral(images_[i].data(), imageSums_[i].data(), width, height);
		sqrIntegral(images_[i].data(), imageSqrSums_[i].data(), width, height);
	}

	std::vector<uint32_t> templSum = sum(templ, channels, tWidth, tHeight);
	std::vector<uint64_t> templSqrSum = sqrSum(templ, channels, tWidth, tHeight);

	float weight = 1.0f / (tWidth * tHeight);
	std::vector<float> templSumScale(channels);
	float templSqrSumScale = 0.0f;

	for (int c = 0; c < channels; c++) {
		templSumScale[c] = weight * templSum[c];
		templSqrSumScale += templSqrSum[c] - weight * templSum[c] * templSum[c];
	}

	matchCCORR(src, dst, channels, width, height, templ, tWidth, tHeight);

	for (int y = 0; y < height - tHeight + 1; y++) {
		for (int x = 0; x < width - tWidth + 1; x++) {
			if (channels == 1) {
				float imgSum = float(
					(imageSums_[0][(y + tHeight) * (width + 1) + x + tWidth] - imageSums_[0][y * (width + 1) + x + tWidth]) - 
					(imageSums_[0][(y + tHeight) * (width + 1) + x] - imageSums_[0][y * (width + 1) + x]));

				float imgSqrSum = float(
					(imageSqrSums_[0][(y + tHeight) * (width + 1) + x + tWidth] - imageSqrSums_[0][y * (width + 1) + x + tWidth]) -
					(imageSqrSums_[0][(y + tHeight) * (width + 1) + x] - imageSqrSums_[0][y * (width + 1) + x]));

				float ccorr = dst[y * (width - tWidth + 1) + x];

				dst[y * (width - tWidth + 1) + x] = (ccorr - imgSum * templSumScale[0]) /
					std::sqrtf(templSqrSumScale * (imgSqrSum - weight * imgSum * imgSum));

			} else if (channels == 3) {
				float imgSum_r = float(
					(imageSums_[0][(y + tHeight) * (width + 1) + x + tWidth] - imageSums_[0][y * (width + 1) + x + tWidth]) -
					(imageSums_[0][(y + tHeight) * (width + 1) + x] - imageSums_[0][y * (width + 1) + x]));

				float imgSum_g = float(
					(imageSums_[1][(y + tHeight) * (width + 1) + x + tWidth] - imageSums_[1][y * (width + 1) + x + tWidth]) -
					(imageSums_[1][(y + tHeight) * (width + 1) + x] - imageSums_[1][y * (width + 1) + x]));

				float imgSum_b = float(
					(imageSums_[2][(y + tHeight) * (width + 1) + x + tWidth] - imageSums_[2][y * (width + 1) + x + tWidth]) -
					(imageSums_[2][(y + tHeight) * (width + 1) + x] - imageSums_[2][y * (width + 1) + x]));

				float imgSqrSum_r = float(
					(imageSqrSums_[0][(y + tHeight) * (width + 1) + x + tWidth] - imageSqrSums_[0][y * (width + 1) + x + tWidth]) -
					(imageSqrSums_[0][(y + tHeight) * (width + 1) + x] - imageSqrSums_[0][y * (width + 1) + x]));

				float imgSqrSum_g = float(
					(imageSqrSums_[1][(y + tHeight) * (width + 1) + x + tWidth] - imageSqrSums_[1][y * (width + 1) + x + tWidth]) -
					(imageSqrSums_[1][(y + tHeight) * (width + 1) + x] - imageSqrSums_[1][y * (width + 1) + x]));

				float imgSqrSum_b = float(
					(imageSqrSums_[2][(y + tHeight) * (width + 1) + x + tWidth] - imageSqrSums_[2][y * (width + 1) + x + tWidth]) -
					(imageSqrSums_[2][(y + tHeight) * (width + 1) + x] - imageSqrSums_[2][y * (width + 1) + x]));

				float num = dst[y * (width - tWidth + 1) + x] - imgSum_r * templSumScale[0] - imgSum_g * templSumScale[1] - imgSum_b * templSumScale[2];
				float den = std::sqrtf(templSqrSumScale * (imgSqrSum_r - weight * imgSum_r * imgSum_r
															+ imgSqrSum_g - weight * imgSum_g * imgSum_g
															+ imgSqrSum_b - weight * imgSum_b * imgSum_b));

				dst[y * (width - tWidth + 1) + x] = num / den;
			}

		}
	}
}