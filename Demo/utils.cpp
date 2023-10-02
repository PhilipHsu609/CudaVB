#define _USE_MATH_DEFINES

#include "utils.h"
#include <cmath>

void rgb2bgr(std::vector<uint8_t> &image) {
	for (size_t i = 0; i < image.size(); i += 3) {
		std::swap(image[i], image[i + 2]);
	}
}

void bgr2rgb(std::vector<uint8_t> &image) {
	rgb2bgr(image);
}

std::vector<float> gaussianKernel(int kernelSize, float sigma) {
	std::vector<float> kernel(kernelSize * kernelSize);

    int kernelRadius = kernelSize >> 1;
    float sum = 0;

	for (int y = -kernelRadius; y <= kernelRadius; y++) {
		for (int x = -kernelRadius; x <= kernelRadius; x++) {
            float r = std::sqrt(x * x + y * y);
            float value = std::exp(-(r * r) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            kernel[(y + kernelRadius) * kernelSize + (x + kernelRadius)] = value;
            sum += value;
        }
    }

    for (int i = 0; i < kernel.size(); i++) {
        kernel[i] /= sum;
    }

    return kernel;

}