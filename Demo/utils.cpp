#define _USE_MATH_DEFINES

#include "EasyBMP/EasyBMP.h"
#include "utils.h"

#include <vector>
#include <cassert>
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

std::vector<uint8_t> loadRAW(BMP &bmp) {
    int width = bmp.TellWidth();
    int height = bmp.TellHeight();
    int channels = bmp.TellBitDepth() / 8;

    std::vector<uint8_t> raw(width * height * channels);

    for (int y = 0; y < height; y++) {
        uint8_t *row = raw.data() + y * width * channels;
        if (channels == 1) {
            bmp.Write8bitRow(row, width * channels, y);
        } else if (channels == 3) {
            bmp.Write24bitRow(row, width * channels, y);
        }
    }

    return raw;
}

void saveRAW(BMP &bmp, std::vector<uint8_t> &raw) {
    int width = bmp.TellWidth();
    int height = bmp.TellHeight();
    int channels = bmp.TellBitDepth() / 8;

    assert(raw.size() == width * height * channels && "Resize bmp before call this function.");

    for (int y = 0; y < height; y++) {
        uint8_t *row = raw.data() + y * width * channels;
        if (channels == 1) {
            bmp.Read8bitRow(row, width * channels, y);
        } else if (channels == 3) {
            bmp.Read24bitRow(row, width * channels, y);
        }
    }
}