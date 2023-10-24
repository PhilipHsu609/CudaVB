#define _USE_MATH_DEFINES

#include "my_cuda_lib.h"
#include "utils.h"
#include "lib_test.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define TEST_CUDA

void testCCL() {
	srand(128);

	int width, height, bytesPerPixel;
	int channels = 1;
	uint8_t *src = stbi_load("./image/lena_gray.bmp", &width, &height, &bytesPerPixel, channels);
	uint8_t *bw = new uint8_t[width * height * channels];
	int *label = new int[width * height * channels];

#ifdef TEST_CUDA
	uint8_t *devSrc{}, *devBW{};
	int *devLabel{};
	cudaAlloc((void **)&devSrc, width * height * channels * sizeof(uint8_t));
	cudaAlloc((void **)&devBW, width * height * channels * sizeof(uint8_t));
	cudaAlloc((void **)&devLabel, width * height * channels * sizeof(int));

	toGPU(src, devSrc, width * height * channels * sizeof(uint8_t));

	// Call kernel
	binarizeCuda(devSrc, devBW, width, height, getThreshVal_OtsuCuda(devSrc, width, height));
	connectedComponentsCuda(devBW, devLabel, width, height);

	toCPU(devLabel, label, width * height * channels * sizeof(int));
#else
	binarizeCpu(src, bw, width, height, getThreshVal_OtsuCpu(src, width, height));
	connectedComponentsCpu(bw, label, width, height);
#endif

	int nLabels = flattenL(label, width * height);
	std::cout << "Number of labels: " << nLabels << std::endl;

	std::vector<std::vector<int>> colorMap(nLabels);
	colorMap[0] = { 0, 0, 0 };
	for (int i = 1; i < nLabels; i++) {
		colorMap[i] = { rand() & 255, rand() & 255, rand() & 255 };
	}

	uint8_t *dst = new uint8_t[width * height * 3];
	for (int i = 0; i < width * height; i++) {
		dst[i * 3] = colorMap[label[i]][0];
		dst[i * 3 + 1] = colorMap[label[i]][1];
		dst[i * 3 + 2] = colorMap[label[i]][2];
	}

#ifdef TEST_CUDA
	stbi_write_bmp("./image/output_ccl_cuda.bmp", width, height, 3, dst);
	cudaRelease(devSrc);
	cudaRelease(devBW);
	cudaRelease(devLabel);
#else
	stbi_write_bmp("./image/output_ccl_cpu.bmp", width, height, 3, dst);
	delete[] bw;
#endif

	stbi_image_free(src);
	delete[] label;
	delete[] dst;
}

void testBilinear() {
	// read image
	int width, height, bpp;
	int channels = 3;
	uint8_t *src = stbi_load("./image/lena_color.bmp", &width, &height, &bpp, channels);

	int dstWidth = width * 2.2;
	int dstHeight = height * 3.3;

	std::vector<uint8_t> dst(dstWidth * dstHeight * channels);
	uint8_t *dstPtr = dst.data();

#ifdef TEST_CUDA
	// allocate memory on GPU
	uint8_t *devSrc{}, *devDst{}; // need to be initialized to nullptr
	cudaAlloc((void **)&devSrc, width * height * channels * sizeof(uint8_t));
	cudaAlloc((void **)&devDst, dstWidth * dstHeight * channels * sizeof(uint8_t));

	// copy data from CPU to GPU
	toGPU(src, devSrc, width * height * channels * sizeof(uint8_t));

	// call kernel
	bilinearCuda(devSrc, devDst, channels, width, height, dstWidth, dstHeight);

	// copy data from GPU to CPU
	toCPU(devDst, dstPtr, dst.size() * sizeof(uint8_t));

	// write image
	stbi_write_bmp("./image/output_bilinear_cuda.bmp", dstWidth, dstHeight, channels, dst.data());

	// free memory
	cudaRelease(devSrc);
	cudaRelease(devDst);
#else
	bilinearCpu(src, dstPtr, channels, width, height, dstWidth, dstHeight);
	stbi_write_bmp("./image/output_bilinear_cpu.bmp", dstWidth, dstHeight, channels, dst.data());
#endif

	stbi_image_free(src);

}

void testConv() {
	// read image
	int width, height, bpp;
	int channels = 3;
	uint8_t *src = stbi_load("./image/lena_color.bmp", &width, &height, &bpp, channels);

	std::vector<uint8_t> dst(width * height * channels);
	uint8_t *dstPtr = dst.data();

	std::vector<float> kernel = gaussianKernel(5, 1.0f);

#ifdef TEST_CUDA
	// allocate memory on GPU
	uint8_t *devSrc{}, *devDst{}; // need to be initialized to nullptr
	cudaAlloc((void **)&devSrc, width * height * channels * sizeof(uint8_t));
	cudaAlloc((void **)&devDst, width * height * channels * sizeof(uint8_t));

	// copy data from CPU to GPU
	toGPU(src, devSrc, width * height * channels * sizeof(uint8_t));

	// call kernel
	conv2DCuda(devSrc, devDst, channels, width, height, kernel.data(), 5);

	// copy data from GPU to CPU
	toCPU(devDst, dstPtr, dst.size() * sizeof(uint8_t));

	// write image
	stbi_write_bmp("./image/output_conv_cuda.bmp", width, height, channels, dst.data());

	// free memory
	cudaRelease(devSrc);
	cudaRelease(devDst);
#else
	conv2DCpu(src, dst.data(), channels, width, height, kernel.data(), 5);
	stbi_write_bmp("./image/output_conv_cpu.bmp", width, height, channels, dst.data());
#endif

	stbi_image_free(src);
}

void testHough() {
	// read image
	int width, height, bpp;
	int channels = 1;
	uint8_t *src = stbi_load("./image/lena_canny.bmp", &width, &height, &bpp, channels);

	int maxLines = 50, numLines;
	std::vector<float> lines(2 * maxLines);
	float *linesPtr = lines.data();

#ifdef TEST_CUDA
	// allocate memory on GPU
	uint8_t *devSrc{}; // need to be initialized to nullptr
	cudaAlloc((void **)&devSrc, width * height * channels * sizeof(uint8_t));

	// copy data from CPU to GPU
	toGPU(src, devSrc, width * height * channels * sizeof(uint8_t));

	// call kernel
	numLines = houghLinesCuda(devSrc, linesPtr, maxLines, width, height, 1, M_PI / 180, 150);

	// free memory
	cudaRelease(devSrc);
#else
	numLines = houghLinesCpu(src, linesPtr, maxLines, width, height, 1, M_PI / 180, 150);
#endif

	std::cout << "Number of lines: " << numLines << std::endl;

	std::vector<float> rho, theta;
	for(int i = 0; i < numLines; i++) {
		rho.push_back(linesPtr[2 * i]);
		theta.push_back(linesPtr[2 * i + 1]);
	}

	printData(rho);
	printData(theta);

	stbi_image_free(src);
}