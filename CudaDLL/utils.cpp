#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "my_cuda_lib.h"

extern "C" int deviceCount();

int deviceCount() {
	int count;
	checkCudaErrors(cudaGetDeviceCount(&count));
	return count;
}