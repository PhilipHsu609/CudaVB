#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "my_cuda_lib.h"

#include "cuda_runtime.h"
#include "math_constants.h"
#include "device_launch_parameters.h"
#include "cooperative_groups.h"
#include "helper_cuda.h"
#include "utils.h"

#include <cmath>

namespace cg = cooperative_groups;

extern "C" int houghLinesCpu(
	const uint8_t *src,
	float *lines,
	int maxLines,
	int width,
	int height,
	float rho,
	float theta,
	int threshold
);

extern "C" int houghLinesCuda(
	const uint8_t *devSrc,
	float *lines,
	int maxLines,
	int width,
	int height,
	float rho,
	float theta,
	int threshold
);

template<int PIXELS_PER_THREAD>
__global__ void buildPointListKernel(const uint8_t *src, int *points, int *counter, int width, int height);
__global__ void linesAccumGlobalKernel(const int *points, int pointsCount, int *accum, float irho, float theta, int aWidth);
__global__ void linesGetResultKernel(const int *accum, float *lines, int maxLines, int aWidth, int aHeight, float rho, float theta, int threshold, int *counter);

int buildPointList(const uint8_t *src, int *points, int *counter, int width, int height);
void linesAccum(const int *points, int pointsCount, int *accum, int aWidth, int aHeight, float rho, float theta);
int linesGetResult(const int *accum, float *lines, int maxLines, int aWidth, int aHeight, float rho, float theta, int threshold, int *counter);

template<int PIXELS_PER_THREAD>
__global__ void buildPointListKernel(const uint8_t *src, int *points, int *counter, int width, int height) {
	__shared__ int sQueue[4][32 * PIXELS_PER_THREAD];
	__shared__ int sQueueCounter[4];
	__shared__ int sGlobalStart[4];
	cg::thread_block g = cg::this_thread_block();

	int x = blockIdx.x * blockDim.x * PIXELS_PER_THREAD + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (threadIdx.x == 0) {
		sQueueCounter[threadIdx.y] = 0;
	}

	g.sync();

	if (y < height) {
		const uint8_t *row = src + y * width;
		for(int i = 0, xx = x; i < PIXELS_PER_THREAD && xx < width; i++, xx += blockDim.x) {
			if (row[xx]) {
				int val = (y << 16) | xx;
				int idx = atomicAdd(sQueueCounter + threadIdx.y, 1);
				sQueue[threadIdx.y][idx] = val;
			}
		}
	}

	g.sync();

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		int totalSize = 0;
		for(int i = 0; i < 4; i++) {
			sGlobalStart[i] = totalSize;
			totalSize += sQueueCounter[i];
		}
		
		int globalOffset = atomicAdd(counter, totalSize);
		for (int i = 0; i < blockDim.y; i++) {
			sGlobalStart[i] += globalOffset;
		}
	}

	g.sync();

	int qSize = sQueueCounter[threadIdx.y];
	int gIdx = sGlobalStart[threadIdx.y] + threadIdx.x;
	for(int i = threadIdx.x; i < qSize; i += blockDim.x, gIdx += blockDim.x) {
		points[gIdx] = sQueue[threadIdx.y][i];
	}
}

int buildPointList(const uint8_t *src, int *points, int *counter, int width, int height) {
	constexpr int PIXELS_PER_THREAD = 16;

	checkCudaErrors(cudaMemset(counter, 0, sizeof(int)));

	dim3 threadsPerBlock(32, 4);
	dim3 numBlocks(divUp(width, threadsPerBlock.x * PIXELS_PER_THREAD), divUp(height, threadsPerBlock.y));

	buildPointListKernel<PIXELS_PER_THREAD><<<numBlocks, threadsPerBlock>>>(src, points, counter, width, height);

	int count;
	checkCudaErrors(cudaMemcpy(&count, counter, sizeof(int), cudaMemcpyDeviceToHost));
	return count;
}

__global__ void linesAccumGlobalKernel(const int *points, int pointsCount, int *accum, float irho, float theta, int aWidth) {
	int n = blockIdx.x;
	float ang = n * theta;

	float sin, cos;
	sincosf(ang, &sin, &cos);
	sin *= irho;
	cos *= irho;

	int shift = (aWidth - 3) / 2;

	int *accRow = accum + (n + 1) * aWidth;
	for (int i = threadIdx.x; i < pointsCount; i += blockDim.x) {
		int val = points[i];
		int x = val & 0xFFFF;
		int y = (val >> 16) & 0xFFFF;

		int r = __float2int_rn(x * cos + y * sin) + shift;
		atomicAdd(accRow + r + 1, 1);
	}
}

void linesAccum(const int *points, int pointsCount, int *accum, int aWidth, int aHeight, float rho, float theta) {
	dim3 threadsPerBlock(1024);
	dim3 numBlocks(aHeight - 2);
	linesAccumGlobalKernel<<<numBlocks, threadsPerBlock>>>(points, pointsCount, accum, 1.0f / rho, theta, aWidth);
}

__global__ void linesGetResultKernel(const int *accum, float *lines, int maxLines, int aWidth, int aHeight, float rho, float theta, int threshold, int *counter) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y * blockDim.y + threadIdx.y;

	if(r >= aWidth - 2 || n >= aHeight - 2) {
		return;
	}

	int votes = accum[(n + 1) * aWidth + r + 1];

	if (votes > threshold &&
		votes > accum[(n + 1) * aWidth + r] &&
		votes >= accum[(n + 1) * aWidth + r + 2] &&
		votes > accum[n * aWidth + r + 1] &&
		votes >= accum[(n + 2) * aWidth + r + 1]) {
		
		float radius = (r - (aWidth - 3) * 0.5f) * rho;
		float angle = n * theta;

		int idx = atomicAdd(counter, 1);
		if (idx < maxLines) {
			lines[idx * 2] = radius;
			lines[idx * 2 + 1] = angle;
		}
	}
}

int linesGetResult(const int *accum, float *lines, int maxLines, int aWidth, int aHeight, float rho, float theta, int threshold, int *counter) {
	checkCudaErrors(cudaMemset(counter, 0, sizeof(int)));

	dim3 threadsPerBlock(32, 8);
	dim3 numBlocks(divUp(aWidth - 2, threadsPerBlock.x), divUp(aHeight - 2, threadsPerBlock.y));
	linesGetResultKernel<<<numBlocks, threadsPerBlock>>>(accum, lines, maxLines, aWidth, aHeight, rho, theta, threshold, counter);

	int linesCount;
	checkCudaErrors(cudaMemcpy(&linesCount, counter, sizeof(int), cudaMemcpyDeviceToHost));
	return std::min(linesCount, maxLines);
}

int houghLinesCuda(
	const uint8_t *devSrc,
	float *lines,
	int maxLines,
	int width,
	int height,
	float rho,
	float theta,
	int threshold
) {
	// References:
	//    https://github.com/opencv/opencv_contrib/blob/5b95c334d7c78778de48948644458bfb8adbb7f7/modules/cudaimgproc/src/hough_lines.cpp#L140
	//    https://github.com/opencv/opencv_contrib/blob/5b95c334d7c78778de48948644458bfb8adbb7f7/modules/cudaimgproc/src/cuda/build_point_list.cu
	//    https://github.com/opencv/opencv_contrib/blob/5b95c334d7c78778de48948644458bfb8adbb7f7/modules/cudaimgproc/src/cuda/hough_lines.cu

	float *devLines;  // result
	int *devPoints;   // point of interest
	int *devCounter;  // temp counter
	int *devAccum;    // accumulator (\theta by \rho)

	int numangle = std::round(CUDART_PI_F / theta);
	int numrho = std::round(((width + height) * 2 + 1) / rho);

	int aWidth = numrho + 2;     // +2 for padding (avoid edge cases in linesGetResult where we will search the local maximum in 4 directions)
	int aHeight = numangle + 2;  // +2 for padding

	checkCudaErrors(cudaMalloc(&devPoints, width * height * sizeof(int)));
	checkCudaErrors(cudaMalloc(&devLines, maxLines * 2 * sizeof(float)));
	checkCudaErrors(cudaMalloc(&devAccum, aWidth * aHeight * sizeof(int)));
	checkCudaErrors(cudaMalloc(&devCounter, sizeof(int)));
	
	// Pick all the non-zero points
	int pointsCount = buildPointList(devSrc, devPoints, devCounter, width, height);

	// Accumulate the lines that pass through the point in the Hough space
	linesAccum(devPoints, pointsCount, devAccum, aWidth, aHeight, rho, theta);

	// Find the lines such that the accumulator has values above the threshold
	int linesCount = linesGetResult(devAccum, devLines, maxLines, aWidth, aHeight, rho, theta, threshold, devCounter);

	checkCudaErrors(cudaMemcpy(lines, devLines, linesCount * 2 * sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(devPoints));
	checkCudaErrors(cudaFree(devCounter));
	checkCudaErrors(cudaFree(devAccum));
	checkCudaErrors(cudaFree(devLines));

	return linesCount;
}

int houghLinesCpu(
	const uint8_t *src,
	float *lines,
	int maxLines,
	int width,
	int height,
	float rho,
	float theta,
	int threshold
) {
	std::vector<int> points;
	for(int y = 0; y < height; y++) {
		const uint8_t *row = src + y * width;
		for(int x = 0; x < width; x++) {
			if (row[x]) {
				points.push_back((y << 16) | x);
			}
		}
	}

	int numangle = std::round(CUDART_PI_F / theta);
	int numrho = std::round(((width + height) * 2 + 1) / rho);

	int aWidth = numrho + 2;
	int aHeight = numangle + 2;
	std::vector<std::vector<int>> accum(aHeight, std::vector<int>(aWidth, 0));

	for(int i = 0; i < points.size(); i++) {
		int val = points[i];
		int x = val & 0xFFFF;
		int y = (val >> 16) & 0xFFFF;

		for(int n = 0; n < numangle; n++) {
			float ang = n * theta;
			float r = x * std::cos(ang) + y * std::sin(ang);
			int ridx = std::round(r / rho) + (aWidth - 3) / 2 + 1;
			accum[n + 1][ridx]++;
		}
	}

	int linesCount = 0;
	for(int n = 0; n < numangle; n++) {
		for(int r = 0; r < numrho; r++) {
			int votes = accum[n + 1][r + 1];
			if (votes > threshold &&
				votes > accum[n + 1][r] &&
				votes >= accum[n + 1][r + 2] &&
				votes > accum[n][r + 1] &&
				votes >= accum[n + 2][r + 1]) {
				
				float radius = (r - (aWidth - 3) * 0.5f) * rho;
				float angle = n * theta;

				if (linesCount < maxLines) {
					lines[linesCount * 2] = radius;
					lines[linesCount * 2 + 1] = angle;
				}
				linesCount++;
			}
		}
	}

	return linesCount;
}