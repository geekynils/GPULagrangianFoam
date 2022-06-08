#ifndef GPU_TRACKING_H
#define GPU_TRACKING_H

#include <vector>

// Facade header file, needs to be included in order to use this code.
#define GPU_PRECISION_DP

// Only use signed integers
typedef int len_t;

// In order to avoid problems with scalar defined by the OpenFOAM framework..
#ifdef GPU_PRECISION_DP
	typedef double gpuScalar;
#else
	typedef float gpuScalar;
#endif

// Constants
const int MAX_FACES_PER_CELL = 6;

/**
 * Static data, either already known or calculated only once, does not change
 * over time.
 */
struct StaticHostData
{
	std::vector<gpuScalar> cellCentres;
	std::vector<int> nFacesPerCell;
	std::vector<int> faceLabelsPerCell;
	std::vector<int> faceLabelsIndex;
	std::vector<int> owners;
	std::vector<int> neighbours;
	std::vector<gpuScalar> faceCentres;
	std::vector<gpuScalar> faceNormals;
	std::vector<gpuScalar> lambdaCnumerator;
	gpuScalar wallImpactDistance;
	gpuScalar dt; // delta t
	int nCells;
	int nParticles; // assumed static for now
};

struct TimeDepHostData
{
	// Particle data
	std::vector<gpuScalar> particlePositions;
	std::vector<gpuScalar> estimatedEndPositions;
	std::vector<gpuScalar> Uparticle;
	std::vector<gpuScalar> diameters;
	std::vector<int> occupancy;

	// Velocity field in the mesh
	std::vector<gpuScalar> U;

	// To calculate
	std::vector<int> nFacesFound;
	std::vector<int> facesFound;
	std::vector<int> facesFoundIndex;

	std::vector<int> particleLabels;

	std::vector<int> particleLabelsRemaining;

	std::vector<gpuScalar> lambdas;
	std::vector<int> facesHit;
	std::vector<gpuScalar> steptFraction;
};

// Exposed Methods

void initializeGPUTracking(
	StaticHostData& staticHostData,
	TimeDepHostData& timeDepHostData
);

void calcLambdas();

void moveParticles();

void cleanUp();

#endif
