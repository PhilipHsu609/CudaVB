#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "my_cuda_lib.h"
#include "utils.h"

#include <vector>
#include <cstdint>
#include <stdexcept>

extern "C" int deviceCount();

int deviceCount() {
	int count;
	checkCudaErrors(cudaGetDeviceCount(&count));
	return count;
}

int borderInterpolate(int p, int len, int borderType) {
	// Border type
	// 0: constant
	// 1: replicate
	// 2: reflect
	// 3: reflect_101

	if ((unsigned)p < (unsigned)len)
		;
	else if (borderType == 1)
		p = p < 0 ? 0 : len - 1;
	else if (borderType == 2 || borderType == 3) {
		int delta = borderType == 3;
		if (len == 1)
			return 0;
		do {
			if (p < 0)
				p = -p - 1 + delta;
			else
				p = len - 1 - (p - len) - delta;
		} while ((unsigned)p >= (unsigned)len);
	} else if (borderType == 0)
		p = -1;
	else
		throw std::runtime_error("Unknown/unsupported border type");
	return p;
}

/*
 Various border types, image boundaries are denoted with '|'

 * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
 * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
 * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
 * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
 *
 * References:
 *     https://github.com/opencv/opencv/blob/f3c724d449b6431c407611fcc5d1ecedfe81e77d/modules/core/src/copy.cpp#L748
 *     https://github.com/opencv/opencv/blob/f3c724d449b6431c407611fcc5d1ecedfe81e77d/modules/core/src/copy.cpp#L798
 *     https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
 */
std::vector<uint8_t> padding2D(
    const uint8_t *src,
    int channels,
    int width,
    int height,
    int top,
    int bottom,
    int left,
    int right,
    int borderType,
    uint8_t value
) {
    // Border type
    // 0: constant
    // 1: replicate
    // 2: reflect
    // 3: reflect_101

    int targetWidth = width + left + right;
    int targetHeight = height + top + bottom;

    std::vector<uint8_t> target(targetWidth * targetHeight * channels, value);

    for (int y = 0; y < height; y++) {
        const uint8_t *src_row = &src[y * width * channels];
        uint8_t *target_row = &target[((y + top) * targetWidth + left) * channels];
        std::copy(src_row, src_row + width * channels, target_row);
    }

    if (borderType == 0)
        return target;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < left; x++) {
            int j = borderInterpolate(x - left, width, borderType) * channels;
            uint8_t *target_row = target.data();
            for (int c = 0; c < channels; c++)
                target_row[((y + top) * targetWidth + x) * channels + c] = target[((y + top) * targetWidth + left) * channels + j + c];
        }
        for (int x = 0; x < right; x++) {
            int j = borderInterpolate(x + width, width, borderType) * channels;
            uint8_t *target_row = target.data();
            for (int c = 0; c < channels; c++)
                target_row[((y + top) * targetWidth + (x + left + width)) * channels + c] = target[((y + top) * targetWidth + left) * channels + j + c];
        }
    }

    for (int y = 0; y < top; y++) {
        int j = borderInterpolate(y - top, height, borderType);
        std::copy(&target[(j + top) * targetWidth * channels], &target[(j + top + 1) * targetWidth * channels], &target[y * targetWidth * channels]);
    }

    for (int y = 0; y < bottom; y++) {
        int j = borderInterpolate(y + height, height, borderType);
        std::copy(&target[(j + top) * targetWidth * channels], &target[(j + top + 1) * targetWidth * channels], &target[(y + top + height) * targetWidth * channels]);
    }

    return target;
}