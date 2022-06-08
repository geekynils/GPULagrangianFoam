#ifndef PARTICLE_DATA_H
#define PARTICLE_DATA_H

#include "cuvector.h"
#include "GPUTracking.h"
#include "cudaMath.h"
#include "CuData.h"

/**
 * Keeps all the time dependent particle related data in together. Data is 
 * stored using the vector types defined by CUDA (vector_functions.h and
 * vector_types.h). 4 Component vectors are used because they can be fetched in
 * one instruction (float) or in two instruction (double). The accessor 
 * functions, however return three component vector types after fetching.
 */
class ParticleData: public CuData
{
protected:

	//!
	//! Position of the particles.
	cuvector<gpuScalar> particlePositions;
	
	//! Estimated end position of the particle, we check weather a face is hit
	//! when traveling from a to b.
	cuvector<gpuScalar> estimatedEndPositions;
	
	//!
	//! Velocity of the particle.
	cuvector<gpuScalar> U;
	
	//!
	//! Diameter of the particle.
	cuvector<gpuScalar> diameters;
	
	//!
	//! Cell label of the cell in which the particle resides.
	cuvector<int> occupancy;

	//!
	//! Holds the number of faces found (per particle) after calculating
	//! lambda_c (eg those where lambda_c is in [0,1].
	cuvector<int> nFacesFound;
	
	//!
	//! Holds the face labels of the faces found, per particle.
	cuvector<int> facesFound;
	
	//! Holds the index where the faces found per particle starts in the
	//! facesFound vector.
	cuvector<int> facesFoundIndex;

	//! After first calculating lambda_c this contains the labels for the which
	//! still need to be tracked. Those with no faces for which lambda_c is in
	//! [0, 1] can be move to the end position without changing occupancy.
	cuvector<int> particleLabels;
	
	//!
	//! Labels of the remaining particles of the tracking step before.
	cuvector<int> particleLabelsRemaining;

	//! Holds the lambda_a calculated.
	cuvector<gpuScalar> lambdas;
	
	//! Holds the face label hit by a particle. -1 if none. Note that
	//! facesFound and facesHit are two different things. FacesFound holds the
	//! face label for which lambda_c in [0,1], facesHit holds the face label
	//! which was hit by the particle (if any).
	cuvector<int> facesHit;
	
	//! For particle traveling over multiple cells this is used to keep track
	//!  of the fraction of the time step already traveled.
	cuvector<gpuScalar> steptFraction;
	
	const int nParticles;
	
	//! After evaluating lambda_c for each particle the set of particles which
	//!  stays in the cell is determined and does not need to be considered any
	//!  further. Variable holds the number of particles which still need to be
	//!  tracked.
	int nParticlesInSet;

	int nRemainingParticles;

	//!
	//! Disallow assignment or copy constructor.
	ParticleData& operator=(const ParticleData&);
	ParticleData(const ParticleData&);

	void checkSize();

public:

	ParticleData(
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
	);

	//- Accesor methods
	
	vec3 getParticlePosition(const int& particleLabel) const;
	
	vec3 getEstimatedEndPosition(const int& particleLabel) const;
	
	vec3 getU(const int& particleLabel) const;
	
	vec3 getParticleDiameter(const int& particleLabel) const;
	
	
	void setParticlePosition(const vec3& position, const int& particleLabel);
	
	void setEstimatedEndPosition(const vec3& endPos, const int& particleLabel);
	
	void setU(const vec3& U, const int& particleLabel);
	
	void setParticleDiameter(const vec3& d, const int& particleLabel);
	
	
	//- Further Methods
	
	//!
	//! Returns the total size of the data in bytes.
	long getDataSize();
};

#include "ParticleDataI.h"

#endif
