#ifndef UTILS_H
#define UTILS_H

#pragma once

#include <iostream>
#include <iterator>
#include <algorithm>

template<typename T>
void printData(T &container) {
	std::for_each(std::begin(container), std::end(container), [](auto &x) {
		std::cout << x << " ";
	});
	std::cout << std::endl;
}

#endif // !UTILS_H