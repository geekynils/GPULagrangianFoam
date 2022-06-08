/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright held by original author
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation; either version 2 of the License, or (at your
    option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM; if not, write to the Free Software Foundation,
    Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA

\*---------------------------------------------------------------------------*/

#include "gpuParticleCloud.H"
#include "fvMesh.H"
#include "volFields.H"
#include "interpolationCellPoint.H"
#include "meshSearch.H"
#include "octree.H"
#include "octreeDataCell.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam {

    defineParticleTypeNameAndDebug(gpuParticle, 0);
    defineTemplateTypeNameAndDebug(Cloud<gpuParticle>, 0);
};

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::gpuParticleCloud::gpuParticleCloud(
    const fvMesh& mesh,
    const word& cloudName,
    bool readFields
)
:
    Cloud<gpuParticle>(mesh, cloudName, false),
    mesh_(mesh),
    particleProperties_(
        IOobject(
            "particleProperties",
            mesh_.time().constant(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    ),
    rhop_(dimensionedScalar(particleProperties_.lookup("rhop")).value()),
    e_(dimensionedScalar(particleProperties_.lookup("e")).value()),
    mu_(dimensionedScalar(particleProperties_.lookup("mu")).value()),
    staticHostData(NULL),
    timeDepHostData(NULL)
{
    if (readFields) {
        gpuParticle::readFields(*this);
    }
}

Foam::gpuParticleCloud::gpuParticleCloud(
	const fvMesh& mesh,
	const word& cloudName,
	IDLList<gpuParticle>& particleList
):
	Cloud<gpuParticle>(mesh, cloudName, false),
	mesh_(mesh),
	particleProperties_ (
		IOobject (
			"particleProperties",
			mesh_.time().constant(),
			mesh_,
			IOobject::MUST_READ,
			IOobject::NO_WRITE
		)
	),
	rhop_(dimensionedScalar(particleProperties_.lookup("rhop")).value()),
	e_(dimensionedScalar(particleProperties_.lookup("e")).value()),
	mu_(dimensionedScalar(particleProperties_.lookup("mu")).value()),
    staticHostData(NULL),
    timeDepHostData(NULL)
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::gpuParticleCloud::move(const dimensionedVector& g) {

    const volScalarField& rho = mesh_.lookupObject<const volScalarField>("rho");
    const volVectorField& U = mesh_.lookupObject<const volVectorField>("U");
    const volScalarField& nu = mesh_.lookupObject<const volScalarField>("nu");

    interpolationCellPoint<scalar> rhoInterp(rho);
    interpolationCellPoint<vector> UInterp(U);
    interpolationCellPoint<scalar> nuInterp(nu);

    gpuParticle::trackData td(*this, rhoInterp, UInterp, nuInterp, g.value());

    Cloud<gpuParticle>::move(td);
}

void Foam::gpuParticleCloud::initGPU(bool validate) {

	this->validate = validate;

	Info << "Flatten and upload static data structures.." << nl;

	assert(staticHostData == NULL);
	assert(timeDepHostData == NULL);

	staticHostData = new StaticHostData();
	timeDepHostData = new TimeDepHostData();

	// Fetching mesh data

	getCellCentres(staticHostData->cellCentres);

	getNfacesPerCell(staticHostData->nFacesPerCell);

	getFaceLabelsPerCell(staticHostData->faceLabelsIndex,
		staticHostData->faceLabelsPerCell);

	getOwnersAndNeighbours(staticHostData->owners, staticHostData->neighbours);

	getFaceData(staticHostData->faceCentres, staticHostData->faceNormals);

	staticHostData->dt = mesh_.time().deltaT().value();

	int nCells = staticHostData->cellCentres.size() / 4;

	staticHostData->nCells = nCells;


	// Fetching particle data

	int nParticles = this->size();

	staticHostData->nParticles = nParticles;

	// TODO Should be for example the radius of the particle
	staticHostData->wallImpactDistance = 0.0005;

	getParticleData(*this,
		timeDepHostData->particlePositions,
		timeDepHostData->Uparticle,
		timeDepHostData->diameters,
		timeDepHostData->occupancy
	);


	// Set the correct sizes for the time dependent vectors
	timeDepHostData->nFacesFound.resize(nParticles);
	timeDepHostData->facesFound.resize(nParticles*MAX_FACES_PER_CELL);
	timeDepHostData->facesFoundIndex.resize(nParticles);
	timeDepHostData->particleLabels.resize(nParticles);
	timeDepHostData->particleLabelsRemaining.resize(nParticles);
	timeDepHostData->lambdas.resize(nParticles);
	timeDepHostData->facesHit.resize(nParticles);

	// Setting correct sizes for yet empty vectors

	staticHostData->lambdaCnumerator.resize(
		staticHostData->faceLabelsPerCell.size()
	);
	timeDepHostData->steptFraction.resize(nParticles);
	timeDepHostData->Uparticle.resize(nParticles*4);
	timeDepHostData->U.resize(nCells*4);
	timeDepHostData->diameters.resize(nParticles);
	timeDepHostData->particlePositions.resize(nParticles*4);
	timeDepHostData->estimatedEndPositions.resize(nParticles*4);

	initializeGPUTracking(*staticHostData, *timeDepHostData);
}


void Foam::gpuParticleCloud::checkParticles() {

	Info << "Validation enabled, checking particles.. " << nl;

	// See genRandCloud.C for comments

    treeBoundBox meshBb(mesh_.points());

	scalar typDim = meshBb.avgDim()/(2.0*Foam::cbrt(scalar(mesh_.nCells())));

	treeBoundBox bb (
		meshBb.min(),
		meshBb.max() + vector(typDim, typDim, typDim)
	);

	octreeDataCell shapes(mesh_);

	octree<octreeDataCell> oc (
		bb,  		// overall bounding box
		shapes,     // all information needed to do checks on cells
		1,          // min. levels
		10.0,       // max. size of leaves
		10.0        // maximum ratio of cubes v.s. cells
	);

	for(
		gpuParticleCloud::iterator iter = this->begin();
		iter != this->end();
		++iter
	){
		gpuParticle& particle = iter();
		// label expectedCell = mesh_.findCell(particle.position());
		label expectedCell = oc.find(particle.position());

		if(expectedCell != particle.cell() ) {

			Info << "Particle claims to be in cell " << particle.cell()
				 << nl;
			Info << "But it's in cell " << expectedCell
				 << nl;
			Info << exit(FatalError);
		}
	}
}

void Foam::gpuParticleCloud::moveGPU() {

	Info << "Flatten and upload time-dependent data structures.." << nl;

	assert(staticHostData->nParticles == this->size());

	const volVectorField& Ufoam = mesh_.lookupObject<const volVectorField>("U");

	flattenVectorField(Ufoam, timeDepHostData->U);

	getParticleData(
		*this,
		timeDepHostData->particlePositions,
		timeDepHostData->Uparticle,
		timeDepHostData->diameters,
		timeDepHostData->occupancy
	);

	if(validate) {
		checkParticles();
	}

	Info << "Invoking CUDA code.." << nl;

	moveParticles();

	// Write back
	updateCloudObject();
}

void Foam::gpuParticleCloud::updateCloudObject() {

	std::vector<gpuScalar>& particlePositions
		= timeDepHostData->particlePositions;
	std::vector<gpuScalar>& U = timeDepHostData->Uparticle;
	std::vector<int>& occupancy = timeDepHostData->occupancy;

	int i=0;

	for (
		gpuParticleCloud::iterator iter = this->begin();
		iter != this->end();
		++iter
	){
        gpuParticle& particle = iter();

        particle.position().x() = particlePositions.at(i*4);
        particle.position().y() = particlePositions.at(i*4 + 1);
        particle.position().z() = particlePositions.at(i*4 + 2);

        particle.U().x() = U.at(i*4);
        particle.U().y() = U.at(i*4 + 1);
        particle.U().z() = U.at(i*4 + 2);

        particle.cell() = occupancy.at(i);

        i++;
	}
}

void Foam::gpuParticleCloud::info() const {

    Info << "Cloud: " << this->name() << nl
         << "    Current number of parcels = "
         << this->size() << nl;
}

// Destructor

Foam::gpuParticleCloud::~gpuParticleCloud() {

	if(staticHostData != NULL)
		delete staticHostData;
	if(timeDepHostData != NULL)
		delete timeDepHostData;
}


// ************************************************************************* //
