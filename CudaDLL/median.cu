#include "my_cuda_lib.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <npp.h>

extern "C" void medianFilter(const uint8_t *src, uint8_t *dst, int channels, int width, int height, int kernelWidth, int kernelHeight) {
	NppiSize oSizeROI = {width, height};
	NppiSize oMaskSize = {kernelWidth, kernelHeight};
	NppiPoint oAnchor = {kernelWidth / 2, kernelHeight / 2};

	Npp32u nBufferSize;
	Npp8u *pBuffer{};

	if (channels == 1) {
		nppiFilterMedianGetBufferSize_8u_C1R(oSizeROI, oMaskSize, &nBufferSize);
		cudaMalloc(&pBuffer, nBufferSize);
		nppiFilterMedian_8u_C1R(src, width, dst, width, oSizeROI, oMaskSize, oAnchor, pBuffer);
	} else if (channels == 3) {
		nppiFilterMedianGetBufferSize_8u_C3R(oSizeROI, oMaskSize, &nBufferSize);
		cudaMalloc(&pBuffer, nBufferSize);
		nppiFilterMedian_8u_C3R(src, width * 3, dst, width * 3, oSizeROI, oMaskSize, oAnchor, pBuffer);
	}

	cudaFree(pBuffer);
}