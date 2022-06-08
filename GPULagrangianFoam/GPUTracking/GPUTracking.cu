#include <cuda.h>
#include <cstdio>
#include <cassert>

#include "GPUTracking.h"
#include "deviceManagement.h"
#include "ParticleEngine.h"

// Global pointers

FlatMesh *mesh = NULL;
ParticleEngine *particleEngine = NULL;

/**
 * Initializes the vector wrappers and sets up the objects for the particle
 * engine.
 */
void initialize(
	StaticHostData& staticHostData,
	TimeDepHostData& timeDepHostData
){

	int nParticles = staticHostData.nParticles;
	int nCells = staticHostData.nCells;

	mesh = new FlatMesh(
		staticHostData.cellCentres,
		staticHostData.nFacesPerCell,
		staticHostData.faceLabelsPerCell,
		staticHostData.faceLabelsIndex,
		staticHostData.owners,
		staticHostData.neighbours,
		staticHostData.faceCentres,
		staticHostData.faceNormals,
		timeDepHostData.U,
		staticHostData.lambdaCnumerator,
		staticHostData.wallImpactDistance,
		staticHostData.dt,
		nCells
	);

	particleEngine = new ParticleEngine(
		timeDepHostData.particlePositions,
		timeDepHostData.estimatedEndPositions,
		timeDepHostData.Uparticle,
		timeDepHostData.diameters,
		timeDepHostData.occupancy,
		timeDepHostData.nFacesFound,
		timeDepHostData.facesFound,
		timeDepHostData.facesFoundIndex,
		timeDepHostData.particleLabels,
		timeDepHostData.particleLabelsRemaining,
		timeDepHostData.lambdas,
		timeDepHostData.facesHit,
		timeDepHostData.steptFraction,
		nParticles,
		*mesh
	);
}

void freeData() {

	if(mesh != NULL) {
		delete mesh;
		mesh = NULL;
	}

	if(particleEngine != NULL) {
		delete particleEngine;
		particleEngine = NULL;
	}
}

// Publicly exposed functions

void initializeGPUTracking(
	StaticHostData& staticHostData,
	TimeDepHostData& timeDepHostData
){
	initCUDA();
	initialize(staticHostData, timeDepHostData);
}

void calcLambdas() {

	particleEngine->calcLambdas();
}

void moveParticles() {
	particleEngine->runStep();
}

void cleanUp() {

	freeData();
}
