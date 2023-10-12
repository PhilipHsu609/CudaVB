#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "my_cuda_lib.h"

extern "C" bool cudaAlloc(void **gpuPtr, size_t size);
extern "C" bool cudaRelease(void **gpuPtr);
extern "C" bool toGPU(void *cpuPtr, void *gpuPtr, size_t size);
extern "C" bool toCPU(void *cpuPtr, void *gpuPtr, size_t size);

bool cudaAlloc(void **gpuPtr, size_t size) {
    if(*gpuPtr != nullptr) {
        return false;
    }
    checkCudaErrors(cudaMalloc(gpuPtr, size));
    return true;
}

bool cudaRelease(void **gpuPtr) {
    if(*gpuPtr == nullptr) {
        return false;
    }
    checkCudaErrors(cudaFree(*gpuPtr));
    return true;
}

bool toGPU(void *cpuPtr, void *gpuPtr, size_t size) {
    if(gpuPtr == nullptr || cpuPtr == nullptr) {
        return false;
    }
    checkCudaErrors(cudaMemcpy(gpuPtr, cpuPtr, size, cudaMemcpyHostToDevice));
    return true;
}

bool toCPU(void *cpuPtr, void *gpuPtr, size_t size) {
    if(gpuPtr == nullptr || cpuPtr == nullptr) {
        return false;
    }
    checkCudaErrors(cudaMemcpy(cpuPtr, gpuPtr, size, cudaMemcpyDeviceToHost));
    return true;
}