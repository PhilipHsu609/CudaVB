#ifndef UTILS_H
#define UTILS_H

#pragma once

#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>

template<typename T>
void printData(T &container) {
	std::for_each(std::begin(container), std::end(container), [](auto &x) {
		std::cout << x << " ";
	});
	std::cout << std::endl;
}

void rgb2bgr(std::vector<uint8_t> &image);
void bgr2rgb(std::vector<uint8_t> &image);

std::vector<float> gaussianKernel(int kernelSize, float sigma);

#endif // !UTILS_H