#include "my_cuda_lib.h"

#include <math_constants.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <cmath>

extern "C" void rotateCuda(const uint8_t *devSrc, uint8_t *devDst, int channels, int width, int height, double angle, double centerX, double centerY) {
	NppiSize oSrcSize{ width, height };
	NppiRect oSrcROI{ 0, 0, width, height };
	NppiRect oDstROI{ 0, 0, width, height };

	NppiInterpolationMode eInterp = NPPI_INTER_LINEAR;

	centerX = (centerX == -1.0) ? width / 2.0 : centerX;
	centerY = (centerY == -1.0) ? height / 2.0 : centerY;

	double rad = angle * CUDART_PI / 180.0;
	double shiftX = (1.0 - std::cos(rad)) * centerX - std::sin(rad) * centerY;
	double shiftY = (1.0 - std::cos(rad)) * centerY + std::sin(rad) * centerX;

	if (channels == 1) {
		nppiRotate_8u_C1R(
			devSrc, oSrcSize, width, oSrcROI,
			devDst, width, oDstROI,
			angle, shiftX, shiftY, eInterp
		);
	} else if (channels == 3) {
		nppiRotate_8u_C3R(
			devSrc, oSrcSize, width * channels, oSrcROI,
			devDst, width * channels, oDstROI,
			angle, shiftX, shiftY, eInterp
		);
	}
}