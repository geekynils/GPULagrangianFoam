#include <cstdio>

#include "Kernels.h"
#include "deviceManagement.h"

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void getKernelConfig(int& nBlocks, int& nThreadsPerBlock, const int nItems,
	const int bs) {

	if(nItems < bs) {
		nBlocks = 1;
		nThreadsPerBlock = nItems;
	} else {
		nThreadsPerBlock = bs;
		nBlocks =  nItems / bs;
		// Create one more block if nCells is not a multiple of the block size
		// threads unnecessary threads need to be disabled.
		if(nItems % nThreadsPerBlock != 0)
			nBlocks++;
	}
}

/**
 * Initializes CUDA, looks for a GPU not used by a display and uses it as a
 * CUDA device.
 */
void initCUDA() {

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Found %i devices.\n", deviceCount);

    for (int devId = 0; devId < deviceCount; ++devId) {

		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, devId);

		if (props.kernelExecTimeoutEnabled > 0) {

			// This device is running a display

		} else {

			// This device is not.
			// To use the first device that is not running a display:
			cudaSetDevice(devId);

			printf("Device %i does not seem to be used for a display.\n",
				devId);
		    printf("Choosing device %d: \"%s\" with Compute %d.%d capability.\n",
				devId, props.name, props.major, props.minor);
		    printf("Processor Count on GPU: %i.\n\n",
		    	props.multiProcessorCount);

			break;
		}
    }

    checkCUDAError("When trying to set the CUDA device");

	cudaFuncSetCacheConfig(calcLambdaAKernel, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(findFacesKernel, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(calcLambdacnumKernel, cudaFuncCachePreferL1);

	checkCUDAError("When trying to set cache preference");
}

__global__ void memsetScalarKernel (
	gpuScalar* const array,
	int num,
	int len
){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= len) return;
	array[i] = num;
}

__global__ void memsetIntegralKernel(
	int* const array,
	int num,
	int len
){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= len) return;
	array[i] = num;
}

// TODO templated memset
void cuMemsetScalar(cuvector<gpuScalar>& a, gpuScalar num) {

	int nBlocks, nThreadsPerBlock;
	getKernelConfig(nBlocks, nThreadsPerBlock, a.size());

	memsetScalarKernel<<<nBlocks, nThreadsPerBlock>>>
		(a.devicePtr(), num, a.size());

	cudaDeviceSynchronize();

	checkCUDAError("When executing cuMemsetScalar");
}

void cuMemsetIntegral(cuvector<int>& a, int num) {

	int nBlocks, nThreadsPerBlock;
	getKernelConfig(nBlocks, nThreadsPerBlock, a.size());

	memsetIntegralKernel<<<nBlocks, nThreadsPerBlock>>>
		(a.devicePtr(), num, a.size());

	cudaDeviceSynchronize();

	checkCUDAError("When executing cuMemsetIntegral");
}
