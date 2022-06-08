#include "fvCFD.H"

#include "gpuParticle.H"
#include "gpuParticleCloud.H"
#include "GPUTracking.h"

#include "serialization.h"

#include <list>
#include <fstream>
#include <iostream>

#include <cstdio>
#include <cassert>
#include <iomanip>

#include <sys/time.h>

// Avoids linker errors
label gpuParticle::instances;

inline scalar fequal(scalar& first, scalar& second) {

	if(abs(first-second)<1e-9)
		return true;
	return false;
}

/**
 * Simple test which reads binary data (current position, destination, lambda
 * and faceHit) calculates for each pair of positions the lambda and figures out
 * through which face (if any) the particle goes. The calculated data is then
 * compare with the data read. This test is used to validate the kernel
 * functions.
 */
int main(int argc, char** argv) {

	Foam::argList::validOptions.insert("data", "dataDirectory");
	argList::validOptions.insert("validate", "");

	#include "setRootCase.H"
	#include "createTime.H"
	#include "createMesh.H"
	
	string dataDir = args.option("data");
	bool validate = args.optionFound("validate");

	// Read expected data ------------------------------------------------------
	
	std::list<vector> particlePositionList;
	std::list<vector> endPositionList;
	std::list<int> occupancy_;
	std::list<int> facesHitExpected;
	std::list<scalar> lambdasExpected;
	
	particlePositionList = readVectorListBinary(dataDir + "/a.data");
	endPositionList = readVectorListBinary(dataDir + "/b.data");
	facesHitExpected = readListBinary<int>(dataDir + "/facesHit.data");
	lambdasExpected = readListBinary<scalar>(dataDir + "/lambdas.data");
	occupancy_ = readListBinary<int>(dataDir + "/cells.data");

	std::vector<scalar> particlePositions = listToVector(particlePositionList);
	std::vector<scalar> particleEndPositions = listToVector(endPositionList);
	std::vector<int> occupancy = listToVector(occupancy_);


	// Get data from OpenFOAM --------------------------------------------------

	IDLList<gpuParticle> particleList;
	gpuParticleCloud cloud(mesh, "particleCloud", particleList);

	StaticHostData staticHostData;

	cloud.getCellCentres(staticHostData.cellCentres);
	cloud.getNfacesPerCell(staticHostData.nFacesPerCell);
	cloud.getFaceLabelsPerCell(staticHostData.faceLabelsIndex,
		staticHostData.faceLabelsPerCell);
	cloud.getOwnersAndNeighbours(staticHostData.owners,
		staticHostData.neighbours);
	cloud.getFaceData(staticHostData.faceCentres, staticHostData.faceNormals);
	int nCells = staticHostData.cellCentres.size() / 4;
	int nParticles = particleEndPositions.size() / 4;
	staticHostData.nCells = nCells;
	staticHostData.nParticles = nParticles;

	TimeDepHostData timeDepHostData;
	timeDepHostData.particlePositions = particlePositions;
	timeDepHostData.estimatedEndPositions = particleEndPositions;
	timeDepHostData.occupancy = occupancy;

	// Set the correct sizes for the time dependent vectors
	timeDepHostData.nFacesFound.resize(nParticles);
	timeDepHostData.facesFound.resize(nParticles*MAX_FACES_PER_CELL);
	timeDepHostData.facesFoundIndex.resize(nParticles);
	timeDepHostData.particleLabels.resize(nParticles);
	timeDepHostData.particleLabelsRemaining.resize(nParticles);
	timeDepHostData.lambdas.resize(nParticles);
	timeDepHostData.facesHit.resize(nParticles);

	staticHostData.lambdaCnumerator.resize(
		staticHostData.faceLabelsPerCell.size());
	timeDepHostData.steptFraction.resize(nParticles);

	// We don't need the velocity field of the mesh or of the particles for this
	// test.
	timeDepHostData.U.resize(nCells*4);
	timeDepHostData.Uparticle.resize(nParticles*4);
	timeDepHostData.diameters.resize(nParticles);

	// Initialize the particle labels vector -----------------------------------
	// This is used to keep track of the particles for which tracking is not
	// done.

	for(int i=0; i<nParticles; i++) {
		timeDepHostData.particleLabels.at(i) = i;
		timeDepHostData.particleLabelsRemaining.at(i) = i;
	}

	// Calculate lambdas on the GPU --------------------------------------------
	
	typedef struct timeval timeval;
	timeval start, end;

	gettimeofday(&start, NULL);

	initializeGPUTracking(staticHostData, timeDepHostData);

	calcLambdas();

	cleanUp();

	gettimeofday(&end, NULL);

	long long totTime = (end.tv_sec - start.tv_sec) * 1000000
		+ (end.tv_usec - start.tv_usec);

	printf("The whole calculation took %llu usec or ~ %f sec.\n",
		totTime, totTime/1000000.0);


	// Compare results ---------------------------------------------------------
	
	if(validate) {

		std::cout << "                                                                                                        Lambdas              facesHit" << nl
				  << "  Id                        a                                          b                            CPU         GPU          CPU   GPU"
				  << std::endl;

		std::cout.setf(std::ios::fixed, std::ios::floatfield);
		std::cout << std::setprecision(8);

		std::list<scalar>::iterator itLambdasExpected = lambdasExpected.begin();
		std::vector<scalar>::iterator itLambdas = timeDepHostData.lambdas.begin();
		std::list<int>::iterator itFacesHitExpected = facesHitExpected.begin();
		std::vector<int>::iterator itFacesHit = timeDepHostData.facesHit.begin();

		int i=0;
		while(itLambdas != timeDepHostData.lambdas.end())
		{
			if(*itFacesHit != *itFacesHitExpected
			  || !fequal(*itLambdas, *itLambdasExpected))
				printf("Different results on the CPU/GPU!\n");

			// id
			std::cout << std::setw(4) << i;

			// particle positions and destinations
			std::cout << std::setw(8) << "[" << particlePositions.at(i*4) << " "
					  << std::setw(8) << particlePositions.at(i*4 + 1) << " "
					  << std::setw(8) << particlePositions.at(i*4 + 2) << "] ";

			std::cout << std::setw(8) << "[" << particleEndPositions.at(i*4) << " "
					  << std::setw(8) << particleEndPositions.at(i*4 + 1) << " "
					  << std::setw(8) << particleEndPositions.at(i*4 + 2) << "] ";

			// lambdas and facesHit
			std::cout << std::setw(18) << *itLambdasExpected
					  << std::setw(12) << *itLambdas;
			std::cout << std::setw(8)  << *itFacesHitExpected
					  << std::setw(6)  << *itFacesHit;
			std::cout << std::endl;

			if(*itFacesHit != *itFacesHitExpected
			  || !fequal(*itLambdas, *itLambdasExpected))
				printf("\n");

			itLambdasExpected++;
			itLambdas++;
			itFacesHitExpected++;
			itFacesHit++;
			i++;
		}

	}

	// Free memory -------------------------------------------------------------

    // TODO

	return 0;
}
