#ifndef FLAT_MESH_H
#define FLAT_MESH_H

#include "FlatMesh.h"
#include "cudaMath.h"
#include "cuvector.h"
#include "CuData.h"


/**
 * Holds the flat mesh representation suitable for the use of the GPU.
 *
 */
class FlatMesh: public CuData
{
	//!
	//! Holds all the cell centres. Indexed by the label of the cell.
	cuvector<gpuScalar> cellCentres;
	
	//!
	//! Holds the number of faces for each cel, indexed by the cell label.
	cuvector<int> nFacesPerCell;
	
	//! Holds the face labels for each cell. Use the faceLabelsIndex to find the
	//! starting point of the cell labels for a given cell.
	cuvector<int> faceLabelsPerCell;
	
	//!
	//! Index to find the face labels given a cell label.
	cuvector<int> faceLabelsIndex;
	
	//!
	//! For each face holds the owner cell.
	cuvector<int> owners;
	
	//!
	//! For all internal faces holds the neighbour cell.
	cuvector<int> neighbours;
	
	//!
	//! Holds all face centres, indexed by the label of the face.
	cuvector<gpuScalar> faceCentres;

	//!
	//! Holds all face normals, indexed by the label of the face.
	cuvector<gpuScalar> faceNormals;

	//!
	//! Velocity field in the mesh.
	cuvector<gpuScalar> U;

	//! Numerator of the formula for lambda_c does not depend on particle data,
	//! therefore it's calculated at the beginning and the stored here.
	cuvector<gpuScalar> lambdacnum;

	//!
	//! Usually the radius of a particle.
	gpuScalar wallImpactDistance;
	
	//!
	//! delta t
	gpuScalar dt;
	
	int nCells;

	void checkSize();


	//!
	//! Calculates the numerator of the equation for lambda_c.
	void calcLambdacnum();

	// Disallow assignment or copy constructor.

	FlatMesh& operator=(const FlatMesh&);
	FlatMesh(const FlatMesh &);

public:

	// Constructor

	FlatMesh(
		std::vector<gpuScalar>& cellCentres,
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
	);

	// Getters

	inline vec3 getCellCentre(const int& cellLabel) const;

	inline int getNFaces(const int& cellLabel) const;

	inline vec3 getFaceCentre(const int& faceLabel) const;

	inline vec3 getFaceNormal(const int& faceLabel) const;

	inline std::vector<int> getFaceLabels(const int& cellLabel) const;


	// Public Methods

	long getDataSize();

	//! Find the adjacent cell given a cell and a face. Returns -1 if the face
	//!  is a boundary face.
	int findAdjacentCell(const int cellLabel, const int faceLabel);

	// Friends

	friend class ParticleEngine;
};

#include "FlatMeshI.h"

#endif
