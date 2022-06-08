#include "gpuParticleCloud.H"

namespace Foam
{

void Foam::gpuParticleCloud::getCellCentres(std::vector<scalar>& Cc) {

	const int nCells = mesh_.nCells();
	Cc.resize(nCells*4);
	const vectorField& cellCentres = mesh_.cellCentres();
	const cellList& cells = mesh_.cells();

	forAll(cells, i) {

		Cc.at(i*4)     = cellCentres[i].x();
		Cc.at(i*4 + 1) = cellCentres[i].y();
		Cc.at(i*4 + 2) = cellCentres[i].z();
	}
}

void Foam::gpuParticleCloud::getNfacesPerCell(std::vector<int>& nFacesPerCell) {

	const int nCells = mesh_.nCells();
	nFacesPerCell.resize(nCells);
	const cellList& cells = mesh_.cells();
	forAll(cells, i) {
		nFacesPerCell.at(i) = cells[i].nFaces();
	}
}

void Foam::gpuParticleCloud::getFaceLabelsPerCell(
	std::vector<int>& faceLabelsIndex,
	std::vector<int>& faceLabels
){
	const int nCells 		 = mesh_.nCells();
	const int nInternalFaces = mesh_.nInternalFaces();
	const int nTotalFaces    = mesh_.nFaces();
	const cellList& cells 	 = mesh_.cells();
	const int nBoundaryFaces = nTotalFaces - nInternalFaces;

	// Faces are stored indexed by cells, we need to store the internal faces
	// twice.
	const int nFacesPerCell = nInternalFaces*2 + nBoundaryFaces;

	faceLabelsIndex.resize(nCells);
	faceLabels.resize(nFacesPerCell);

	int cellLength = 0;
	int cellStart = 0;
	int nFaces = 0;
	label faceLabel;
	int j=0;

	forAll(cells, i) {

		cellStart += cellLength;
		faceLabelsIndex.at(i) = cellStart;
		nFaces = cells[i].nFaces();

		for(j=0; j<nFaces; j++) {

			faceLabel = cells[i][j];
			faceLabels.at(cellStart + j) = faceLabel;
		}

		cellLength = j;
	}
}

void Foam::gpuParticleCloud::getFaceData(
	std::vector<scalar>& Cf,
	std::vector<scalar>& Sf
){
	const vectorField& Sf_ 	= mesh_.faceAreas();
	const vectorField& Cf_ 	= mesh_.faceCentres();
	const label nFaces 		= mesh_.nFaces();

	Cf.resize(nFaces*4);
	Sf.resize(nFaces*4);

	for(int i=0; i<nFaces; i++) {

		Cf.at(i*4) 	   = Cf_[i].x();
		Cf.at(i*4 + 1) = Cf_[i].y();
		Cf.at(i*4 + 2) = Cf_[i].z();

		Sf.at(i*4) 	   = Sf_[i].x();
		Sf.at(i*4 + 1) = Sf_[i].y();
		Sf.at(i*4 + 2) = Sf_[i].z();
	}
}

void Foam::gpuParticleCloud::getOwnersAndNeighbours (
	std::vector<int>& owner,
	std::vector<int>& neighbour
){
	const labelList& faceOwner = mesh_.faceOwner();
	const labelList& faceNeighbour = mesh_.faceNeighbour();

	assert(mesh_.faces().size() == faceOwner.size());

	owner.resize(faceOwner.size());
	neighbour.resize(mesh_.faceNeighbour().size());

	for(int i=0; i<faceOwner.size(); i++) {
		owner.at(i) = faceOwner[i];
	}

	for(int i=0; i<faceNeighbour.size(); i++) {
		neighbour.at(i) = faceNeighbour[i];
	}
}

}
