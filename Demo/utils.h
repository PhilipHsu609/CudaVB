#ifndef UTILS_H
#define UTILS_H

#pragma once

#include <iostream>
#include <iterator>
#include <algorithm>
#include <chrono>
#include <vector>

class Timer {
public:
    Timer() : m_beg{ clock_t::now() } {}
    void reset() {
        m_beg = clock_t::now();
    }
    double elapsed() const {
        return std::chrono::duration_cast<second_t>(clock_t::now() - m_beg).count();
    }

private:
    using clock_t = std::chrono::high_resolution_clock;
    using second_t = std::chrono::duration<double, std::ratio<1>>;

    std::chrono::time_point<clock_t> m_beg;
};

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