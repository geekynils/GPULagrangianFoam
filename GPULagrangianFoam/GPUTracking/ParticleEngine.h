#ifndef PARTICLE_ENGINE_H
#define PARTICLE_ENGINE_H

#include "FlatMesh.h"
#include "ParticleData.h"
#include "cuvector.h"
#include "GPUTracking.h"
#include "cudaMath.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>
#include <thrust/count.h>

/**
 * Implements particle tracking as described in the paper by Graham B.
 * Macpherson, Niklas Nordin and Henry G. Weller. "Particle tracking in
 * unstructured, arbitrary polyhedral meshes for use in CFD and molecular
 * dynamics." - 2008.
 *
 * The calculation of lambda_c and lambda_a is executed on the GPU.
 *
 */
class ParticleEngine: public ParticleData {

	FlatMesh& mesh;
	
	//! Reduces the set of particles which still need tracking to those which
	//!  hit a face.
	void reduceParticles();

	void reduceParticlesDevice();

	//! Calculates lambda_a. Invokes the concerning CUDA Kernel.
	void calcLambdaA();

	//! Calculates lambda_c for each particle and determines which particles
	//!  are to be considered. Invokes the concerning CUDA Kernel.
	void findFaces();

	//! Estimates the end pos of the particle using it's position and velocity.
	void estimateEndPos();

	//! Based on the result of calcLambdas() (input: facesHit and lambdas) this
	//! method will move the particle to the face hit (if any) and change
	//! update the occupancy information of the particles.
	void moveParticles();

	//! Updates the velocity of a particle using the velocity field in the
	//!  mesh.
	vec3 updateVelocity(const vec3& Uparticle, const vec3& Ufield) const;

	//! Specular reflection. Reflects the particle in the same way as a
	//! football is reflected when kicked against a wall.
	//!  References
	//!  - http://mathworld.wolfram.com/Reflection.html
	//!  - http://stackoverflow.com/questions/4430170/question-on-specular-reflection
	vec3 reflect(const vec3 U, const int faceLabel) const;

public:

	ParticleEngine(
		std::vector<gpuScalar>& particlePositions,
		std::vector<gpuScalar>& estimatedEndPositions,
		std::vector<gpuScalar>& U,
		std::vector<gpuScalar>& diameters,
		std::vector<int>& occupancy,
		std::vector<int>& nFacesFound,
		std::vector<int>& facesFound,
		std::vector<int>& facesFoundIndex,
		std::vector<int>& particleLabels,
		std::vector<int>& particleLabelsBefore,
		std::vector<gpuScalar>& lambdas,
		std::vector<int>& facesHit,
		std::vector<gpuScalar>& steptFraction,
		int nParticles,
		FlatMesh& mesh
	);

	//!
	//! Calculates lambda_a and finds the faces hit by all particles. Executes
	//! findFaces() and the calcLambdaA() for the concerning particles.
	void calcLambdas();

	//! Starts with estimating the end position then tracking, moving and
	//! changing occupancy and finally updating the particle's veolcity.
	void runStep();
};

#include "ParticleEngineI.h"

#endif
