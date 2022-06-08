#include <cassert>
#include <cstdio>

#include "FlatMesh.h"
#include "deviceManagement.h"
#include "Kernels.h"

// Constructor
FlatMesh::FlatMesh(
	std::vector<gpuScalar>& cellCentres_,
	std::vector<int>& nFacesPerCell,
	std::vector<int>& faceLabelsPerCell,
	std::vector<int>& faceLabelsIndex,
	std::vector<int>& owners,
	std::vector<int>& neighbours,
	std::vector<gpuScalar>& faceCentres,
	std::vector<gpuScalar>& faceNormals,
	std::vector<gpuScalar>& U,
	std::vector<gpuScalar>& lambdaCnumerator,
	gpuScalar wallImpactDistance,
	gpuScalar dt,
	int nCells
):
	cellCentres(cellCentres_),
	nFacesPerCell(nFacesPerCell),
	faceLabelsPerCell(faceLabelsPerCell),
	faceLabelsIndex(faceLabelsIndex),
	owners(owners),
	neighbours(neighbours),
	faceCentres(faceCentres),
	faceNormals(faceNormals),
	U(U),
	lambdacnum(lambdaCnumerator),
	wallImpactDistance(wallImpactDistance),
	dt(dt),
	nCells(nCells)
{
	checkSize();

	this->cellCentres.upload();
	this->nFacesPerCell.upload();
	this->faceLabelsPerCell.upload();
	this->faceLabelsIndex.upload();
	this->owners.upload();
	this->neighbours.upload();
	this->faceCentres.upload();
	this->faceNormals.upload();
	this->U.upload();

	printf("Allocated %li bytes (~ %f MB) on device for mesh data.\n",
			getDataSize(), getDataSize()/1024.0/1024.0);

	checkCUDAError("When calling FlatMesh::initializeDevice()");

	calcLambdacnum();
}

// Public methods

int FlatMesh::findAdjacentCell(const int cellLabel, const int faceLabel) {

	int adjacentCellLabel;

	if(faceLabel >= static_cast<int>(neighbours.size())) {
		return -1;
	} else {
		// Is this cell the owner of the face?
		if(owners.at(faceLabel) == cellLabel) {
			adjacentCellLabel = neighbours[faceLabel];
		} else {
			adjacentCellLabel = owners[faceLabel];
		}
	}

	return adjacentCellLabel;
}

// Private methods

void FlatMesh::calcLambdacnum() {

	int nBlocks, nThreadsPerBlock;

	getKernelConfig(nBlocks, nThreadsPerBlock, nCells);

	calcLambdacnumKernel<<<nBlocks, nThreadsPerBlock>>> (
		(vec4*)cellCentres.devicePtr(),
		nFacesPerCell.devicePtr(),
		faceLabelsPerCell.devicePtr(),
		faceLabelsIndex.devicePtr(),
		(vec4*)faceCentres.devicePtr(),
		(vec4*)faceNormals.devicePtr(),
		nCells,
		lambdacnum.devicePtr()
	);

	cudaDeviceSynchronize();

	checkCUDAError("Error when executing the calcLambdaCNumeratorKernel");
}


void FlatMesh::checkSize() {

	assert(static_cast<int>(cellCentres.size()) == nCells*4);
	assert(static_cast<int>(nFacesPerCell.size()) == nCells);

	// Same num face normals and face centres
	assert(faceCentres.size() == faceNormals.size());

	// Every face has an owner cell, therefore nOwners == nFaces
	assert(faceNormals.size()/4 == owners.size());

	assert(nCells - 1
		<= static_cast<int>(neighbours.size())
		<= static_cast<int>(faceNormals.size()/4)
	);

	assert(static_cast<int>(U.size()) == nCells*4);

	assert(nCells
			< static_cast<int>(faceLabelsPerCell.size())
		   <= nCells * MAX_FACES_PER_CELL
	);

	assert(faceLabelsPerCell.size() == lambdacnum.size());
}

long FlatMesh::getDataSize() {

	long size = 0;
	size += cellCentres.dataSize();
	size += nFacesPerCell.dataSize();
	size += faceLabelsIndex.dataSize();
	size += owners.dataSize();
	size += neighbours.dataSize();
	size += faceCentres.dataSize();
	size += faceNormals.dataSize();
	size += U.dataSize();
	return size;
}
