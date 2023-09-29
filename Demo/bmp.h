#ifndef BMP_H
#define BMP_H

#pragma once

#include <vector>
#include <string>
#include <windows.h>

const std::string LENA_COLOR{ "lena_color.bmp" };
const std::string LENA_GRAY{ "lena_gray.bmp" };

struct BMP {
	BITMAPFILEHEADER fileHeader;
	BITMAPINFOHEADER infoHeader;
	std::vector<uint8_t> image;

	size_t width() const { return infoHeader.biWidth; }
	size_t height() const { return infoHeader.biHeight; }
	size_t step() const { return infoHeader.biBitCount / 8; }
	size_t stride() const { return width() * step(); }
};

void printBMPHeader(const BMP &bmp);
std::vector<uint8_t> getColorPalette();

BMP loadBMP(const std::string &file);
void saveBMP(const std::string &file, const BMP &bmp);

BMP loadLenaGray();
BMP loadLenaColor();

#endif // !BMP_H