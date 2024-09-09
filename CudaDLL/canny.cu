#include <iostream>

#include "my_cuda_lib.h"
#include "cuda_runtime.h"
#include "npp.h"
#include "helper_cuda.h"

extern "C" void cannyEdgeCuda(const uint8_t *src, uint8_t *dst, int width, int height, int lowThresh, int highThresh) {
	NppiSize oSrcSize{ width, height };
	NppiPoint oSrcOffset{ 0, 0 };

	NppiSize oSizeROI{ width, height };

	int nBufferSize;
	Npp8u *pBuffer{};

	nppiFilterCannyBorderGetBufferSize(oSizeROI, &nBufferSize);

	cudaMalloc(&pBuffer, nBufferSize);

	Npp16s nLowThreshold = lowThresh;
	Npp16s nHighThreshold = highThresh;

	nppiFilterCannyBorder_8u_C1R(
		src, width * sizeof(uint8_t), oSrcSize, oSrcOffset,
		dst, width * sizeof(uint8_t), oSizeROI, NPP_FILTER_SOBEL,
		NPP_MASK_SIZE_3_X_3, nLowThreshold, nHighThreshold, nppiNormL2,
		NPP_BORDER_REPLICATE, pBuffer);

	cudaFree(pBuffer);
}

extern "C" unsigned char foundAryMaxCuda(const uint8_t *src, int width, int height) {
	NppiSize oSizeROI{ width, height };

	int nBufferSize;
	Npp8u *pBuffer{};

	Npp8u MaxValue;

	checkNPPErrors(nppiMaxGetBufferHostSize_8u_C1R(oSizeROI, &nBufferSize));

	checkCudaErrors(cudaMalloc(&pBuffer, nBufferSize));

	checkNPPErrors(nppiMax_8u_C1R(src, width * sizeof(uint8_t), oSizeROI, pBuffer, &MaxValue));

	checkCudaErrors(cudaFree(pBuffer));

	return MaxValue;
}