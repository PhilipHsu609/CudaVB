#include "my_cuda_lib.h"
#include "utils.h"
#include "bmp.h"

#include <vector>
#include <iostream>
#include <cstdint>

int main() {
	BMP bmp = loadBMP("FLAG_B24.BMP");
	BMP g{ loadLenaGray() };

	std::vector<uint8_t> dst(bmp.image.size() / 3);
	
	bgr2grayCuda(bmp.image.data(), dst.data(), bmp.width(), bmp.height());

	BMP out;
	out.fileHeader = g.fileHeader;
	out.infoHeader = g.infoHeader;
	out.image = dst;

	out.fileHeader.bfOffBits += 256 * 4;
	out.infoHeader.biWidth = bmp.width();
	out.infoHeader.biHeight = bmp.height();

	saveBMP("FLAG_GRAY.BMP", out);

	return 0;
}