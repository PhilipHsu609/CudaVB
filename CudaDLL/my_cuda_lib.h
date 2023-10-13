#ifndef MY_CUDA_LIB_H
#define MY_CUDA_LIB_H

#pragma once

#define DLL_EXPORT __declspec(dllexport)

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cstdint>

/*
	* Get the number of CUDA devices
*/
extern "C" DLL_EXPORT int deviceCount();

/*
	* GPU memory management
*/
extern "C" DLL_EXPORT bool cudaAlloc(void **gpuPtr, size_t size);
extern "C" DLL_EXPORT bool cudaRelease(void **gpuPtr);
extern "C" DLL_EXPORT bool toGPU(void *cpuPtr, void *gpuPtr, size_t size);
extern "C" DLL_EXPORT bool toCPU(void *cpuPtr, void *gpuPtr, size_t size);

/*
	* Binarize image
	* 
	* Note: src is a gray scale image
	* Note: threshold is in range [0, 255]
*/
extern "C" DLL_EXPORT void binarizeCpu(const uint8_t *src, uint8_t *dst, int width, int height, uint8_t threshold);
extern "C" DLL_EXPORT void binarizeCuda(const uint8_t *src, uint8_t *dst, int width, int height, uint8_t threshold);

/*
	* Convert RGB to Gray
*/
extern "C" DLL_EXPORT void rgb2grayCpu(const uint8_t *src, uint8_t *dst, int width, int height);
extern "C" DLL_EXPORT void rgb2grayCuda(const uint8_t *src, uint8_t *dst, int width, int height);


/*
	* Convert RGB to HSV
	*
	* Note: For HSV, H range is [0, 180], S range is [0, 255], V range is [0, 255] (OpenCV)
*/
extern "C" DLL_EXPORT void rgb2hsvCpu(const uint8_t *src, uint8_t *dst, int width, int height);
extern "C" DLL_EXPORT void rgb2hsvCuda(const uint8_t *src, uint8_t *dst, int width, int height);

/*
	* Convolution
	* 
	* Note: Kernel size must be odd
*/
extern "C" DLL_EXPORT void conv2DCpu(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const float *kernel, int kernelSize);
extern "C" DLL_EXPORT void conv2DCuda(const uint8_t *src, uint8_t *dst, int channels, int width, int height, const float *kernel, int kernelSize);

/*
	* Dilation
	*
	* Note: Kernel is a 0/1 square matrix
*/
extern "C" DLL_EXPORT void dilateCpu(const uint8_t * src, uint8_t * dst, int channels, int width, int height, const int *kernel, int kernelSize);
extern "C" DLL_EXPORT void dilateCuda(const uint8_t * src, uint8_t * dst, int channels, int width, int height, const int *kernel, int kernelSize);

/*
	* Erosion
	*
	* Note: Kernel is a 0/1 square matrix
*/
extern "C" DLL_EXPORT void erodeCpu(const uint8_t * src, uint8_t * dst, int channels, int width, int height, const int *kernel, int kernelSize);
extern "C" DLL_EXPORT void erodeCuda(const uint8_t * src, uint8_t * dst, int channels, int width, int height, const int *kernel, int kernelSize);

/*
	* Gaussian pyramid
*/
extern "C" DLL_EXPORT void pyrUpCpu(const uint8_t * src, uint8_t * dst, int channels, int width, int height);
//extern "C" DLL_EXPORT void pyrUpCuda(const uint8_t * src, uint8_t * dst, int channels, int width, int height);

extern "C" DLL_EXPORT void pyrDownCpu(const uint8_t * src, uint8_t * dst, int channels, int width, int height);
//extern "C" DLL_EXPORT void pyrDownCuda(const uint8_t * src, uint8_t * dst, int channels, int width, int height);

/*
	* Resize image
	*
	* TODO: Implement bilinear interpolation with CUDA texture memory
*/
extern "C" DLL_EXPORT void bilinearCpu(const uint8_t * src, uint8_t * dst, int channels, int width, int height, int dstWidth, int dstHeight);
extern "C" DLL_EXPORT void bilinearCuda(const uint8_t * src, uint8_t * dst, int channels, int width, int height, int dstWidth, int dstHeight);

/*
	* Histogram equalization
*/
extern "C" DLL_EXPORT void equalizeHistCPU(const uint8_t * src, uint8_t * dst, int channels, int width, int height);
extern "C" DLL_EXPORT void equalizeHistCuda(const uint8_t * src, uint8_t * dst, int channels, int width, int height);

#endif // !MY_CUDA_LIB_H