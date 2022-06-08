#include <cstdio>
#include <cuda.h>

//#define DEBUG
#include "cudaAssert.h"
#include "eclipse.h"
#include "GPUTracking.h"
#include "cudaMath.h"
#include "deviceCodeHelpers.h"

__device__ gpuScalar lambda_c(gpuScalar& nom, vec3& b, vec3& Cc, vec3& Sf);

__global__ void findFacesKernel(

	// Static data
	int *particleLabels,
	int *nFacesPerCell,
	int *faceLabelsPerCell,
	int *faceLabelsIndex,
	vec4 *SfPtr,
	vec4 *CcPtr,

	// Time dependent data
	int *occupancy,
	vec4 *posPtr,
	gpuScalar *numeratorPtr,
	int *facesFoundPtr,
	int *nFacesFoundPtr,
	int nParticles
);

__global__ void calcLambdacnumKernel (
	vec4 *cellCentres,
	int *nFacesPerCell,
	int *faceLabelsPerCell,
	int *faceLabelsIndex,
	vec4 *CfPtr,
	vec4 *SfPtr,
	int nCells,
	gpuScalar* lambdaCnumerator
);


__global__ void calcLambdaAKernel (
	vec4 *a_, vec4 *b_,
	vec4 *Cf_, vec4 *Sf_,
	int *particleLabels,
	int *nFacesFound_,
	int *facesFoundPtr,
	int remainingParticles,
	int nNeighbours,
	gpuScalar wallImpactDistance,
	int *facesHitPtr,
	gpuScalar *lambdasPtr
);

__global__ void reorderNFacesFound(
	const int * const particleLabels,
	const int * const nFacesFound,
	int * const nFacesFoundReordered,
	const int nLabels
);
