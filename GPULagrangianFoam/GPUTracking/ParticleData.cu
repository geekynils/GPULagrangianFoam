#include "ParticleData.h"
#include <cassert>
#include <cstdio>

// Constructor

// TODO Naming: Having the same name for the constructor argument as for the
//      instance variable seems to work. Need to check what the standard says
//      about this.

ParticleData::ParticleData(
	std::vector<gpuScalar>& particlePositions,
	std::vector<gpuScalar>& estimatedEndPositions,
	std::vector<gpuScalar>& U,
	std::vector<gpuScalar>& diameters,
	std::vector<int>& occupancy,
	std::vector<int>& nFacesFound,
	std::vector<int>& facesFound,
	std::vector<int>& facesFoundIndex,
	std::vector<int>& particleLabels,
	std::vector<int>& particleLabelsRemaining,
	std::vector<gpuScalar>& lambdas,
	std::vector<int>& facesHit,
	std::vector<gpuScalar>& steptFraction,
	int nParticles
) :
	particlePositions(particlePositions),
	estimatedEndPositions(estimatedEndPositions),
	U(U),
	diameters(diameters),
	occupancy(occupancy),
	nFacesFound(nFacesFound),
	facesFound(facesFound),
	facesFoundIndex(facesFoundIndex),
	particleLabels(particleLabels),
	particleLabelsRemaining(particleLabelsRemaining),
	lambdas(lambdas),
	facesHit(facesHit),
	steptFraction(steptFraction),
	nParticles(nParticles),
	nParticlesInSet(nParticles),
	nRemainingParticles(nParticles)
{
	checkSize();

	printf("Allocated %li bytes (~ %f MB) on device for particle data.\n",
			getDataSize(), getDataSize()/1024.0/1024.0);
}

// TODO Comparison between signed and unsigned int.
void ParticleData::checkSize() {

	assert(static_cast<int>(particlePositions.size()) == nParticles*4);
	assert(static_cast<int>(estimatedEndPositions.size()) == nParticles*4);
	assert(static_cast<int>(U.size()) == nParticles*4);
	assert(static_cast<int>(diameters.size()) == nParticles);
	assert(static_cast<int>(occupancy.size()) == nParticles);
	assert(static_cast<int>(nFacesFound.size()) == nParticles);
	assert(static_cast<int>(facesFound.size()) == nParticles * MAX_FACES_PER_CELL);
	assert(static_cast<int>(facesFoundIndex.size()) == nParticles);
	assert(static_cast<int>(particleLabels.size()) == nParticles);
	assert(static_cast<int>(lambdas.size()) == nParticles);
	assert(static_cast<int>(facesHit.size()) == nParticles);
	assert(static_cast<int>(steptFraction.size()) == nParticles);
}

long ParticleData::getDataSize() {

	long size = 0;
	size += particlePositions.dataSize();
	size += estimatedEndPositions.dataSize();
	size += U.dataSize();
	size += diameters.dataSize();
	size += occupancy.dataSize();
	size += nFacesFound.dataSize();
	size += facesFound.dataSize();
	size += facesFoundIndex.dataSize();
	size += particleLabels.dataSize();
	size += lambdas.dataSize();
	size += facesHit.dataSize();
	size += steptFraction.dataSize();
	return size;
}
