#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "my_cuda_lib.h"

extern "C" bool cudaAlloc(void **gpuPtr, size_t size);
extern "C" bool cudaRelease(void *gpuPtr);
extern "C" bool toGPU(void *cpuPtr, void *gpuPtr, size_t size);
extern "C" bool toCPU(void *cpuPtr, void *gpuPtr, size_t size);

#ifdef DEBUG
#include <iostream>
#endif

bool cudaAlloc(void **gpuPtr, size_t size) {
    if(*gpuPtr != nullptr) {
        return false;
    }
    checkCudaErrors(cudaMalloc(gpuPtr, size));

#ifdef DEBUG
    std::cout << "Malloc address: " << *gpuPtr << "\n";
#endif

    return true;
}

bool cudaRelease(void *gpuPtr) {
    if(gpuPtr == nullptr) {
        return false;
    }
    checkCudaErrors(cudaFree(gpuPtr));
    return true;
}

bool toGPU(void *cpuPtr, void *gpuPtr, size_t size) {
    if(gpuPtr == nullptr || cpuPtr == nullptr) {
        return false;
    }

#ifdef DEBUG
    std::cout << "CPU PTR: " << cpuPtr << "\n";
    std::cout << "GPU PTR: " << gpuPtr << "\n";
    std::cout << "Hey I got ";
    for(int i = 0; i < 10; i++) {
		std::cout << (int)((uint8_t*)cpuPtr)[i] << " ";
	}
    std::cout << "from cpu.\n";
#endif

    checkCudaErrors(cudaMemcpy(gpuPtr, cpuPtr, size, cudaMemcpyHostToDevice));
    return true;
}

bool toCPU(void *gpuPtr, void *cpuPtr, size_t size) {
    if(gpuPtr == nullptr || cpuPtr == nullptr) {
        return false;
    }
    checkCudaErrors(cudaMemcpy(cpuPtr, gpuPtr, size, cudaMemcpyDeviceToHost));

#ifdef DEBUG
    std::cout << "CPU PTR: " << cpuPtr << "\n";
    std::cout << "GPU PTR: " << gpuPtr << "\n";

    std::cout << "Hey I got ";
    for (int i = 0; i < 10; i++) {
        std::cout << (int)((uint8_t *)cpuPtr)[i] << " ";
    }
    std::cout << "from gpu.\n";
#endif

    return true;
}