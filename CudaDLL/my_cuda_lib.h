#ifndef MY_CUDA_LIB_H
#define MY_CUDA_LIB_H

#pragma once

#include <cstdint>

#define DLL_EXPORT __declspec(dllexport)

/*
	* Get number of CUDA devices
	* return: number of CUDA devices
*/
extern "C" DLL_EXPORT int deviceCount();

extern "C" DLL_EXPORT void addWithCuda(int *c, const int *a, const int *b, unsigned int size);


/*
	* Convert BGR to Gray
	* src: input image
	* dst: output image
	* width: image width
	* height: image height
	* 
	* Note: src and dst must be allocated before calling this function
*/
extern "C" DLL_EXPORT void bgr2grayCpu(const uint8_t * src, uint8_t * dst, const size_t width, const size_t height);
extern "C" DLL_EXPORT void bgr2grayCuda(const uint8_t * src, uint8_t * dst, const size_t width, const size_t height);


/*
	* Convert BGR to HSV
	* src: input image
	* dst: output image
	* width: image width
	* height: image height
	* 
	* Note: src and dst must be allocated before calling this function
	* Note: For HSV, H range is [0, 180], S range is [0, 255], V range is [0, 255] (OpenCV)
*/
extern "C" DLL_EXPORT void bgr2hsvCpu(const uint8_t * src, uint8_t * dst, const size_t width, const size_t height);
extern "C" DLL_EXPORT void bgr2hsvCuda(const uint8_t * src, uint8_t * dst, const size_t width, const size_t height);

#endif // !MY_CUDA_LIB_H