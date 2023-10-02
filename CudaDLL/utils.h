#ifndef UTILS_H
#define UTILS_H

#pragma once

#include <vector>
#include <cstdint>

int deviceCount();

int borderInterpolate(int p, int len, int borderType);

std::vector<uint8_t> padding2D(
	const uint8_t *src,
	int channels,
	int width,
	int height,
	int top,
	int bottom,
	int left,
	int right,
	int borderType = 3,
	uint8_t value = 0
);

#endif // !UTILS_H