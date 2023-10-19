#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "my_cuda_lib.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "utils.h"

#include <vector>

extern "C" void connectedComponentsCpu(const uint8_t *src, int *label, int width, int height);
extern "C" void connectedComponentsCuda(const uint8_t *devSrc, int *devLabel, int width, int height);

__global__ void initLabeling(int *label, int width, int height);
__global__ void merge(const uint8_t *src, int *label, int width, int height);
__global__ void compression(int *label, int width, int height);
__global__ void finalLabeling(const uint8_t *src, int *label, int width, int height);

template<typename T>
__device__ __forceinline__ int hasBit(T x, int bit);
__device__ int find(int *label, int x);
__device__ void unionSet(int *label, int x, int y);

int findRoot(const int *P, int i);
void setRoot(int *P, int i, int r);
int set_union(int *P, int i, int j);

template<typename T>
__device__ __forceinline__ int hasBit(T x, int bit) {
	return (x >> bit) & 1;
}

__device__ int find(int *label, int x) {
	int y = x;
	while (label[x] != x) {
		x = label[x];
		label[y] = x;
	}
	return x;
}

__device__ void unionSet(int *label, int x, int y) {
	bool done = false;
	do {
		x = find(label, x);
		y = find(label, y);

		if (x < y) {
			int old = atomicMin(&label[y], x);
			done = (old == y);
			y = old;
		} else if (x > y) {
			int old = atomicMin(&label[x], y);
			done = (old == x);
			x = old;
		} else {
			done = true;
		}
	} while(!done);
}

__global__ void initLabeling(int *label, int width, int height) {
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	int idx = y * width + x;
	
	if(x >= width || y >= height)
		return;

	label[idx] = idx;
}

__global__ void merge(const uint8_t *src, int *label, int width, int height) {
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	int idx = y * width + x;

	if(x >= width || y >= height)
		return;

	unsigned int P = 0;
	
	if (src[idx])
		P |= 0x777;
	if (x + 1 < width && src[idx + 1])
		P |= (0x777 << 1);
	if (y + 1 < height && src[idx + width])
		P |= (0x777 << 4);

	if (x == 0)
		P &= 0xEEEE;

	if (x + 1 >= width)
		P &= 0x3333;
	else if(x + 2 >= width)
		P &= 0x7777;

	if(y == 0)
		P &= 0xFFF0;
	if(y + 1 >= height)
		P &= 0xFF;

	if (P > 0) {
		if(hasBit(P, 0) && src[idx - width - 1])
			unionSet(label, idx, idx - 2 * width - 2);  // top left

		if ((hasBit(P, 1) && src[idx - width]) || (hasBit(P, 2) && src[idx - width + 1]))
			unionSet(label, idx, idx - 2 * width);      // top center

		if (hasBit(P, 3) && src[idx - width + 2])
			unionSet(label, idx, idx - 2 * width + 2);  // top right

		if ((hasBit(P, 4) && src[idx - 1]) || (hasBit(P, 8) && src[idx + width - 1]))
			unionSet(label, idx, idx - 2);              // left
	}
}

__global__ void compression(int *label, int width, int height) {
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	int idx = y * width + x;

	if(x >= width || y >= height)
		return;

	find(label, idx);
}

__global__ void finalLabeling(const uint8_t *src, int *label, int width, int height) {
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	int idx = y * width + x;

	if(x >= width || y >= height)
		return;

	int l = label[idx] + 1;

	if(src[idx])
		label[idx] = l;
	else
		label[idx] = 0;

	if (x + 1 < width) {
		if (src[idx + 1])
			label[idx + 1] = l;
		else
			label[idx + 1] = 0;

		if (y + 1 < height) {
			if(src[idx + width + 1])
				label[idx + width + 1] = l;
			else
				label[idx + width + 1] = 0;
		}
	}

	if (y + 1 < height) {
		if(src[idx + width])
			label[idx + width] = l;
		else
			label[idx + width] = 0;
	}
}

void connectedComponentsCuda(const uint8_t *devSrc, int *devLabel, int width, int height) {
	// References:
	//     https://github.com/opencv/opencv_contrib/blob/4.x/modules/cudaimgproc/src/cuda/connectedcomponents.cu
	//     https://github.com/prittt/YACCLAB/blob/master/cuda/src/labeling_allegretti_2019_BUF.cu
	//     https://ieeexplore.ieee.org/document/8798895

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(divUp((width + 1) / 2, threadsPerBlock.x), divUp((height + 1) / 2, threadsPerBlock.y));

	initLabeling<<<numBlocks, threadsPerBlock>>>(devLabel, width, height);

	merge<<<numBlocks, threadsPerBlock>>>(devSrc, devLabel, width, height);

	compression<<<numBlocks, threadsPerBlock>>>(devLabel, width, height);

	finalLabeling<<<numBlocks, threadsPerBlock>>>(devSrc, devLabel, width, height);
}

int findRoot(const int *P, int i) {
	int root = i;
	while (P[root] < root) {
		root = P[root];
	}
	return root;
}

void setRoot(int *P, int i, int r) {
	while (P[i] < i) {
		int j = P[i];
		P[i] = r;
		i = j;
	}
	P[i] = r;
}

int set_union(int *P, int i, int j) {
	int root = findRoot(P, i);
	if (i != j) {
		int rootj = findRoot(P, j);
		if (root > rootj) {
			root = rootj;
		}
		setRoot(P, j, root);
	}
	setRoot(P, i, root);
	return root;
}

void connectedComponentsCpu(const uint8_t *src, int *label, int width, int height) {
	// References:
	//     https://github.com/opencv/opencv/blob/6e4280ea81b59c6dca45bb9801b758377beead55/modules/imgproc/src/connectedcomponents.cpp#L2354

	const size_t Plength = (size_t(width) * size_t(height) + 1) / 2 + 1;
	std::vector<int> P_(Plength, 0);
	int *P = P_.data();
	int nLabels = 1;

	const int w = width;
	const int h = height;
	for (int r = 0; r < h; r++) {
		const uint8_t * const img_row = src + r * w;
		const uint8_t * const img_row_prev = src + (r - 1) * w;

		int * const imgLabels_row = label + r * w;
		int * const imgLabels_row_prev = label + (r - 1) * w;

		for (int c = 0; c < w; ++c) {

#define condition_p c>0 && r>0 && img_row_prev[c - 1]>0
#define condition_q r>0 && img_row_prev[c]>0
#define condition_r c < w - 1 && r > 0 && img_row_prev[c + 1] > 0
#define condition_s c > 0 && img_row[c - 1] > 0
#define condition_x img_row[c] > 0

			if (condition_x) {
				if (condition_q) {
					//x <- q
					imgLabels_row[c] = imgLabels_row_prev[c];
				} else {
					// q = 0
					if (condition_r) {
						if (condition_p) {
							// x <- merge(p,r)
							imgLabels_row[c] = set_union(P, imgLabels_row_prev[c - 1], imgLabels_row_prev[c + 1]);
						} else {
							// p = q = 0
							if (condition_s) {
								// x <- merge(s,r)
								imgLabels_row[c] = set_union(P, imgLabels_row[c - 1], imgLabels_row_prev[c + 1]);
							} else {
								// p = q = s = 0
								// x <- r
								imgLabels_row[c] = imgLabels_row_prev[c + 1];
							}
						}
					} else {
						// r = q = 0
						if (condition_p) {
							// x <- p
							imgLabels_row[c] = imgLabels_row_prev[c - 1];
						} else {
							// r = q = p = 0
							if (condition_s) {
								imgLabels_row[c] = imgLabels_row[c - 1];
							} else {
								//new label
								imgLabels_row[c] = nLabels;
								P[nLabels] = nLabels;
								nLabels = nLabels + 1;
							}
						}
					}
				}
			} else {
				//x is a background pixel
				imgLabels_row[c] = 0;
			}
		}
	}
#undef condition_p
#undef condition_q
#undef condition_r
#undef condition_s
#undef condition_x

	for (int i = 1; i < nLabels; i++) {
		P[i] = findRoot(P, i);
	}

	for (int r = 0; r < h; r++) {
		int *img_row_start = label + r * w;
		int * const img_row_end = img_row_start + w;
		for (int c = 0; img_row_start != img_row_end; img_row_start++, c++) {
			*img_row_start = P[*img_row_start];
		}
	}
}