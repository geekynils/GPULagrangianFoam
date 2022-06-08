#ifndef DEVICE_MANAGEMENT_H
#define DEVICE_MANAGEMENT_H

#include "GPUTracking.h"
#include "cudaMath.h"
#include "cuvector.h"

void checkCUDAError(const char *msg);

/**
 * Initializes CUDA, looks for a GPU not used by a display and uses it as a
 * CUDA device.
 */
void initCUDA();

/**
 * Sets all values in memory to the value of the given scalar.
 */
void cuMemsetScalar(cuvector<gpuScalar>& a, gpuScalar num);

/**
 * Sets all values in memory to the value of the given integer.
 */
void cuMemsetIntegral(cuvector<int>& a, int num);

/**
 * Calculates the block size and the number of blocks for a given kernel.
 */
void getKernelConfig(int& nBlocks, int& nThreadsPerBlock, const int nItems,
	const int bs=512);

#endif
