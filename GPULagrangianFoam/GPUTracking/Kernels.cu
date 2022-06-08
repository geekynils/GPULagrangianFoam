#include "Kernels.h"
#include "eclipse.h"
#include "cudaAssert.h"

inline __device__ gpuScalar lambdaCFunct(gpuScalar& nom, vec3& b, vec3& Cc, vec3& Sf)
{
	gpuScalar denom = dotP(b - Cc, Sf);

    // check if trajectory is parallel to face
	if(fabs(denom) < SMALL) {

		if(denom < 0.0) {
			denom = -SMALL;
		} else {
			denom = SMALL;
		}
	}

	return nom / denom;
}

// TODO Code duplication
inline __device__ gpuScalar lambdaAFunct(vec3& a, vec3& b, vec3& Cf, vec3& Sf)
{
	gpuScalar nom = dotP(Cf - a, Sf);
	gpuScalar denom = dotP(b - a, Sf);

    // check if trajectory is parallel to face
	if(fabs(denom) < SMALL) {

		if(denom < 0.0) {
			denom = -SMALL;
		} else {
			denom = SMALL;
		}
	}

	return nom / denom;
}

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
)
{
	checkRun
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(!(idx < nParticles))
		return;

	int particleLabel = particleLabels[idx];

	// Fetch data and set pointers
	int cellLabel = occupancy[particleLabel];
	vec3 Cc = getVec3(CcPtr, cellLabel);
	vec3 pos = getVec3(posPtr, particleLabel);
	nFacesFoundPtr += particleLabel;
	facesFoundPtr += particleLabel * MAX_FACES_PER_CELL;

	int *faceLabels = faceLabelsPerCell + faceLabelsIndex[cellLabel];
	int nFaces = nFacesPerCell[cellLabel];
	numeratorPtr += faceLabelsIndex[cellLabel];

    int nFacesFound = 0;
    int faceLabel;
	gpuScalar lambda, numerator;
    vec3 Sf;

    for(int i=0; i<nFaces; i++) {

    	// Fetch data per face
    	faceLabel = faceLabels[i];
    	cudaAssert(faceLabel >= 0);
    	Sf = getVec3(SfPtr, faceLabel);
    	Sf = normVec(Sf);
    	numerator = numeratorPtr[i];

    	lambda = lambdaCFunct(numerator, pos, Cc, Sf);

    	info("%i pos [%f %f %f] Cc [%f %f %f] Sf [%f %f %f] numerator %f lambda_c %f\n",
			particleLabel,
			pos.x, pos.y, pos.z,
			Cc.x, Cc.y, Cc.z,
			Sf.x, Sf.y, Sf.z,
			numerator,
			lambda);

        if(lambda > 0 && lambda < 1) {
        	facesFoundPtr[nFacesFound] = faceLabel;
        	nFacesFound++;
        }
    }

    // Write back
    *nFacesFoundPtr = nFacesFound;
}

__global__ void calcLambdacnumKernel(
	vec4 *cellCentres,
	int *nFacesPerCell,
	int *faceLabelsPerCell,
	int *faceLabelsIndex,
	vec4 *CfPtr,
	vec4 *SfPtr,
	int nCells,
	gpuScalar* lambdacnum
)
{
	checkRun
	int cellLabel = threadIdx.x + blockIdx.x * blockDim.x;
	if(!(cellLabel < nCells))
		return;

	// Fetch cell data and set pointers
	vec3 Cc = getVec3(cellCentres, cellLabel);
	int nFaces = nFacesPerCell[cellLabel];
	faceLabelsPerCell += faceLabelsIndex[cellLabel];
	lambdacnum        += faceLabelsIndex[cellLabel];

	for(int i=0; i<nFaces; i++) {

		// Fetch face data
		int faceLabel = faceLabelsPerCell[i];
		vec3 Cf = getVec3(CfPtr, faceLabel);
		vec3 Sf = getVec3(SfPtr, faceLabel);
		Sf = normVec(Sf);

		gpuScalar numerator = dotP(Cf - Cc, Sf);

		// Write back
		lambdacnum[i] = numerator;
	}
}

__global__ void calcLambdaAKernel (
	vec4 *aPtr,
	vec4 *bPtr,
	vec4 *CfPtr,
	vec4 *SfPtr,
	int *particleLabels,
	int *nFacesFoundPtr,
	int *facesFoundPtr,
	int nRemainingParticles,
	int nNeighbours,
	gpuScalar wallImpactDistance,
	int * const facesHitPtr,
	gpuScalar * const lambdasPtr
)
{
	checkRun
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(!(i < nRemainingParticles)) return;

	int particleLabel = particleLabels[i];

	cudaAssert(particleLabel >= 0);

	int nFacesFound = nFacesFoundPtr[particleLabel];
	facesFoundPtr += particleLabel * MAX_FACES_PER_CELL;

	vec3 a = getVec3(aPtr, particleLabel);
	vec3 b = getVec3(bPtr, particleLabel);

	// Ensure that it does not calculate lambda_ for particles for which no
	// faces were found with lambda_c in [0,1]
	cudaAssert(nFacesFound >= 1);

	int faceLabel=-1, faceHit=-1;
	gpuScalar lambda, smallestLambda=2;
	vec3 Cf, Sf;

	// Usual case one face of the cell was hit.
	if(nFacesFound == 1) {

		faceLabel = *facesFoundPtr;
		Cf = getVec3(CfPtr, faceLabel);
		Sf = getVec3(SfPtr, faceLabel);

		normVec(Sf);

		// Check if we hit a boundary face. Sf points outside of the
		// computational domain. Therefore we move the face centre in the
		// opposite direction.
		// TODO Ensure that distance Cc - Cf > wallImpactDistance

		if(faceLabel >= nNeighbours) {
			Cf = Cf - wallImpactDistance * Sf;
		}

		lambda = lambdaAFunct(a, b, Cf, Sf);

		//cudaAssert(lambda >= 0 && lambda <= 1);

		if(!(lambda >= 0 && lambda <= 1)) {
			info("Lambda not in desired interval!\n"
					"Particle label: %i\n"
					"Face label: %i\n"
					"Lambda: %f\n",
					particleLabel, faceLabel, lambda);
		}

		lambdasPtr[particleLabel] = lambda;
		facesHitPtr[particleLabel] = faceLabel;

		info("Particle %i: found lambda %f, the face %i was hit.\n",
			particleLabel, lambda, faceLabel);

	} else {

		info("Less likely case for particle (more then one face found) %i\n",
			particleLabel);

		// Less likely case two or more faces of the cell were hit.
		for(int i=0; i<nFacesFound; i++) {

			faceLabel = *facesFoundPtr;
			Cf = getVec3(CfPtr, faceLabel);
			Sf = getVec3(SfPtr, faceLabel);

			lambda = lambdaAFunct(a, b, Cf, Sf);

			//cudaAssert(lambda >= 0 && lambda <= 1);

			if(!(lambda >= 0 && lambda <= 1)) {
				info("Lambda not in desired interval!\n"
						"Particle label: %i\n"
						"Face label: %i\n"
						"Lambda: %f\n",
						particleLabel, faceLabel, lambda);
			}

			if(lambda < smallestLambda) {
				smallestLambda = lambda;
				faceHit = faceLabel;
			}

			facesFoundPtr++;
		}

		lambdasPtr[particleLabel] = smallestLambda;
		facesHitPtr[particleLabel] = faceHit;
	}
}


__global__ void reorderNFacesFound(
	const int * const particleLabels,
	const int * const nFacesFound,
	int * const nFacesFoundReordered,
	const int nLabels
) {

	checkRun
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(!(i < nLabels)) return;

	int particleLabel = particleLabels[i];
	nFacesFoundReordered[i] = nFacesFound[particleLabel];

	// printf("thread: %i particleLabel: %i: nFacesFound: %i\n",
	//     i, particleLabel, nFacesFound[particleLabel]);
}
