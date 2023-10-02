#include "bmp.h"

#include <Windows.h>
#include <fstream>
#include <iostream>
#include <cstdint>

void printBMPHeader(const BMP &bmp) {
	// print out bmp headers
	std::cout << "File Header:" << std::endl;
	std::cout << "bfType: " << bmp.fileHeader.bfType << std::endl;
	std::cout << "bfSize: " << bmp.fileHeader.bfSize << std::endl;
	std::cout << "bfReserved1: " << bmp.fileHeader.bfReserved1 << std::endl;
	std::cout << "bfReserved2: " << bmp.fileHeader.bfReserved2 << std::endl;
	std::cout << "bfOffBits: " << bmp.fileHeader.bfOffBits << std::endl << std::endl;

	std::cout << "Info Header:" << std::endl;
	std::cout << "biSize: " << bmp.infoHeader.biSize << std::endl;
	std::cout << "biWidth: " << bmp.infoHeader.biWidth << std::endl;
	std::cout << "biHeight: " << bmp.infoHeader.biHeight << std::endl;
	std::cout << "biPlanes: " << bmp.infoHeader.biPlanes << std::endl;
	std::cout << "biBitCount: " << bmp.infoHeader.biBitCount << std::endl;
	std::cout << "biCompression: " << bmp.infoHeader.biCompression << std::endl;
	std::cout << "biSizeImage: " << bmp.infoHeader.biSizeImage << std::endl;
	std::cout << "biXPelsPerMeter: " << bmp.infoHeader.biXPelsPerMeter << std::endl;
	std::cout << "biYPelsPerMeter: " << bmp.infoHeader.biYPelsPerMeter << std::endl;
	std::cout << "biClrUsed: " << bmp.infoHeader.biClrUsed << std::endl;
	std::cout << "biClrImportant: " << bmp.infoHeader.biClrImportant << std::endl << std::endl;
}

std::vector<uint8_t> getColorPalette() {
	std::vector<uint8_t> palette(256 * 4);
	for (int i= 0; i < 256; i++) {
		palette[4 * i] = palette[4 * i + 1] = palette[4 * i + 2] = i;
		palette[4 * i + 3] = 0;
	}
	return palette;
}

BMP loadBMP(const std::string &filename) {
	BMP bmp;
	std::ifstream ifile(filename, std::ios::binary);
	if (!ifile.is_open()) {
		std::cerr << "Error: " << filename << " not found." << std::endl;
		exit(-1);
	}

	ifile.read((char *)&bmp.fileHeader, sizeof(BITMAPFILEHEADER));
	ifile.read((char *)&bmp.infoHeader, sizeof(BITMAPINFOHEADER));

	ifile.seekg(bmp.fileHeader.bfOffBits, ifile.beg);

	bmp.dataVec.resize(bmp.width() * bmp.height() * bmp.step());

	uint32_t padding = 4 - bmp.stride() % 4;
	for (int y = bmp.height() - 1; y >= 0; y--) {
		ifile.read((char *)(bmp.dataVec.data() + y * bmp.stride()), bmp.stride());
		if (padding != 4) {
			ifile.seekg(padding, ifile.cur);
		}
	}

	ifile.close();

	return bmp;
}

void saveBMP(const std::string &filename, const BMP &bmp) {
	std::ofstream ofile(filename, std::ios::binary);
	if (!ofile.is_open()) {
		std::cerr << "Create " << filename << " failed." << std::endl;
		exit(-1);
	}

	ofile.write((char *)&bmp.fileHeader, sizeof(BITMAPFILEHEADER));
	ofile.write((char *)&bmp.infoHeader, sizeof(BITMAPINFOHEADER));

	if (bmp.infoHeader.biBitCount == 8) {
		std::vector<uint8_t> palette = getColorPalette();
		ofile.write((char *)palette.data(), palette.size());
	}

	ofile.seekp(bmp.fileHeader.bfOffBits, ofile.beg);

	uint32_t padding = 4 - bmp.stride() % 4;
	for (int y = bmp.height() - 1; y >= 0; y--) {
		ofile.write((char *)(bmp.dataVec.data() + y * bmp.stride()), bmp.stride());
		if (padding != 4) {
			ofile.seekp(padding, ofile.cur);
		}
	}

	ofile.close();
}

BMP loadLenaGray() {
	return loadBMP(LENA_GRAY);
}

BMP loadLenaColor() {
	return loadBMP(LENA_COLOR);
}