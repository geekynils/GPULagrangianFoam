#include "ParticleEngine.h"
#include "deviceManagement.h"
#include "Kernels.h"

// Constructor

ParticleEngine::ParticleEngine(
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
	int nParticles,
	FlatMesh& mesh
):
	ParticleData(
		particlePositions,
		estimatedEndPositions,
		U,
		diameters,
		occupancy,
		nFacesFound,
		facesFound,
		facesFoundIndex,
		particleLabels,
		particleLabelsRemaining,
		lambdas,
		facesHit,
		steptFraction,
		nParticles
	),
	mesh(mesh)
{
}

// Public member functions

void ParticleEngine::calcLambdas() {

	particleLabels.upload();
	particlePositions.upload();
	estimatedEndPositions.upload();
	occupancy.upload();

	checkCUDAError("When uploading a, b and occupancy");

	findFaces();

	// Would be required for reducing the particle set on the CPU.
	// nFacesFound.download();
	// facesFound.download();

	checkCUDAError("When downloading the faces found.");

	// reduceParticles();

	reduceParticlesDevice();

	// TODO Smaller set, we don't always need to set the whole set here..
	cuMemsetScalar(lambdas, 2);
	cuMemsetIntegral(facesHit, -1);

	// Special case in which all particles stay in their cell. Happens with the
	// test case which has just one particle.
	if(nRemainingParticles == 0) {
	    for(int i=0; i<nParticles; i++) {
	    	facesHit.at(i) = -1;
			lambdas.at(i) = 2;
		}
		return;
	}

	// Would be required when reducing the particle set on the CPU.
	// particleLabelsRemaining.upload();

	particleLabelsRemaining.download();

	calcLambdaA();

	lambdas.download();
	facesHit.download();
}

void ParticleEngine::runStep() {

	// (Re-) initialize data for new step.
	// TODO use std::fill or memset
	for(int i=0; i<nParticles; i++) {
		steptFraction.at(i) = 0;
	}

	for(int i=0; i<nParticles; i++) {
		particleLabelsRemaining.at(i) = i;
	}

	for(int i=0; i<nParticles; i++) {
		particleLabels.at(i) = i;
	}

	for(int i=0; i<nParticles; i++) {
		nFacesFound.at(i) = 0;
	}

	for(int i=0; i<static_cast<int>(facesFound.size()); i++) {
		facesFound.at(i) = -1;
	}

	nParticlesInSet = nParticles;
	nRemainingParticles = nParticles;

	while(nParticlesInSet > 0) {

		estimateEndPos();

		calcLambdas();

		moveParticles();
	}
}

/**
 * Do not confuse with the moveParticles method in the GPUTracking interface.
 */
void ParticleEngine::moveParticles() {

	for(int i=0; i<nParticlesInSet; i++) {

		int particleLabel = particleLabels[i];

		const vec3 b = getEstimatedEndPosition(particleLabel);
		const int faceHit = facesHit.at(particleLabel);
		const gpuScalar lambda = lambdas.at(particleLabel);

		vec3 Uparticle = getU(particleLabel);
		int cellLabel = occupancy.at(particleLabel);
		vec3 a = getParticlePosition(particleLabel);
		gpuScalar steptFrac = steptFraction.at(particleLabel);

		if(faceHit == -1) {

			// In case no face was hit we move the particle to the end position.
			a = b;
			// steptFrac = 1;

		} else {

			int adjacentCell = mesh.findAdjacentCell(cellLabel, faceHit);

			// Check if a wall was hit.
			if(adjacentCell == -1) {

				// Move the particle onto the wall hit.
				// TODO overload +=
				a = a + lambda * (b - a);

				// Update u so that it points into the reflected direction.
				Uparticle = reflect(Uparticle, faceHit);

				setU(Uparticle, particleLabel);

			} else {

				// Move the particle onto the face hit.
				a = a + lambda * (b - a);

				// Update occupancy
				cellLabel = adjacentCell;

				// Update the velocity taking into a account the field in the
				// new cell.
				vec3 Ufield = getVec3(mesh.U, cellLabel);
				Uparticle = updateVelocity(Uparticle, Ufield);
			}

			// Finally update steptFraction
			steptFrac += (1 - steptFrac) * lambda;
		}

		// Write changes back
		setParticlePosition(a, particleLabel);
		setU(Uparticle, particleLabel);
		occupancy.at(particleLabel) = cellLabel;
		steptFraction.at(particleLabel) = steptFrac;
	}

	// Update the particles in the set to track.
	nParticlesInSet = nRemainingParticles;

	for(int i=0; i<nParticlesInSet; i++) {
		particleLabels.at(i) = particleLabelsRemaining.at(i);
	}
}

// Private member functions

vec3 ParticleEngine::reflect(const vec3 U, const int faceLabel) const {

	vec3 Sf, Ureflected;

	Sf = getVec3(mesh.faceNormals, faceLabel);

	// TODO Norm all surface normals at the beginning.
	Sf = normVec(Sf);

	Ureflected = U - 2 * dotP(U, Sf) * Sf;

	// info("Calculated reflection for u [%f %f %f] at face %i: [%f %f %f]\n",
	//      u.x, u.y, u.z, faceLabel, ureflected.x, ureflected.y, ureflected.z);

	return Ureflected;
}

void ParticleEngine::calcLambdaA() {

	int nBlocks, nThreadsPerBlock;
	getKernelConfig(nBlocks, nThreadsPerBlock, nRemainingParticles);

	calcLambdaAKernel<<<nBlocks, nThreadsPerBlock>>> (
		(vec4*)particlePositions.devicePtr(),
		(vec4*)estimatedEndPositions.devicePtr(),
		(vec4*)mesh.faceCentres.devicePtr(),
		(vec4*)mesh.faceNormals.devicePtr(),
		particleLabelsRemaining.devicePtr(),
		nFacesFound.devicePtr(),
		facesFound.devicePtr(),
		nRemainingParticles,
		mesh.neighbours.size(),
		mesh.wallImpactDistance,
		facesHit.devicePtr(),
		lambdas.devicePtr()
	);

	checkCUDAError("When executing calcLambdaAKernel");
}

void ParticleEngine::reduceParticles() {

	nRemainingParticles = 0;

	int particleLabel;

	for(int i=0; i<nParticlesInSet; i++) {

		particleLabel = particleLabels.at(i);

		if(nFacesFound.at(particleLabel) == 0) {
			continue;
		}

		particleLabelsRemaining.at(nRemainingParticles) = particleLabel;

		nRemainingParticles++;
	}
}

void ParticleEngine::reduceParticlesDevice() {

	// First we need to reorder nFacesFound, so that the number stands at the
	// same position as the particleLabel.

	thrust::device_vector<int> nFacesFoundReordered(nParticlesInSet);

	int nBlocks, nThreadsPerBlock;

	getKernelConfig(nBlocks, nThreadsPerBlock, nParticlesInSet);

	reorderNFacesFound<<<nBlocks, nThreadsPerBlock>>>(
		particleLabels.devicePtr(),
		nFacesFound.devicePtr(),
		raw_pointer_cast(&nFacesFoundReordered[0]),
		nParticlesInSet
	);

	checkCUDAError("When executing reorderNFacesFound");


	// Now we count the number of zeros in the nFacesFound array, then we can
	// calculate the number of remaining particles.

	int nParticlesDone = thrust::count(
		nFacesFoundReordered.begin(),
		nFacesFoundReordered.end(),
		0
	);

	nRemainingParticles = nParticlesInSet - nParticlesDone;

	// It seems that thrust can just sort in place, so we just copy over the
	// contents of particleLabels to particleLabelsRemaining.

	cudaMemcpy(
		particleLabelsRemaining.devicePtr(),
		particleLabels.devicePtr(),
		sizeof(int) * nParticlesInSet,
		cudaMemcpyDeviceToDevice
	);


	thrust::device_ptr<int> particleLabelsRemainingPtr(
		particleLabelsRemaining.devicePtr()
	);

	// Now we sort particle labels descending using nFacesFoundReordered as key.
	// Particle labels for which no faces was found (nFacesFound == 0) will be
	// at the end.

	thrust::sort_by_key(
		nFacesFoundReordered.begin(),
		nFacesFoundReordered.end(),
		particleLabelsRemainingPtr,
		thrust::greater<int>()	// sort descending
	);

}

// TODO GPU implementation
void ParticleEngine::estimateEndPos() {

	for(int i=0; i<nParticlesInSet; i++) {

		const int particleLabel = particleLabels.at(i);
		const gpuScalar dt = mesh.dt;
		const gpuScalar steptFrac = steptFraction.at(particleLabel);

		vec3 a = getParticlePosition(particleLabel);
		vec3 b = getEstimatedEndPosition(particleLabel);
		vec3 u = getU(particleLabel);

		b = a + (1 - steptFrac) * dt * u;

		setEstimatedEndPosition(b, particleLabel);
	}
}

void ParticleEngine::findFaces() {

	int nBlocks, nThreadsPerBlock;

	getKernelConfig(nBlocks, nThreadsPerBlock, nParticlesInSet);

	findFacesKernel<<<nBlocks, nThreadsPerBlock>>> (
		particleLabels.devicePtr(),
		mesh.nFacesPerCell.devicePtr(),
		mesh.faceLabelsPerCell.devicePtr(),
		mesh.faceLabelsIndex.devicePtr(),
		(vec4*)mesh.faceNormals.devicePtr(),
		(vec4*)mesh.cellCentres.devicePtr(),
		occupancy.devicePtr(),
		(vec4*)estimatedEndPositions.devicePtr(),
		mesh.lambdacnum.devicePtr(),
		facesFound.devicePtr(),
		nFacesFound.devicePtr(),
		nRemainingParticles
	);

	cudaDeviceSynchronize();

	checkCUDAError("When finding faces");
}
