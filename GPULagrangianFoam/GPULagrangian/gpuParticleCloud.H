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

Class
    Foam::gpuParticleCloud

Description
    A Cloud of solid particles

SourceFiles
    gpuParticleCloudI.H
    gpuParticleCloud.C
    gpuParticleCloudIO.C

\*---------------------------------------------------------------------------*/

#ifndef gpuParticleCloud_H
#define gpuParticleCloud_H

// OpenFOAM headers
#include "Cloud.H"
#include "gpuParticle.H"
#include "IOdictionary.H"

#include "gpuLagrangianMisc.H"

// C++ headers
#include <vector>
#include <cassert>

// Header to use the CUDA library.
#include "GPUTracking.h"

namespace Foam
{

// Forward declaration of classes
class fvMesh;

/*---------------------------------------------------------------------------*\
                           Class gpuParticleCloud Declaration
\*---------------------------------------------------------------------------*/

class gpuParticleCloud
:
    public Cloud<gpuParticle>
{
        const fvMesh& mesh_;

        IOdictionary particleProperties_;

        scalar rhop_;
        scalar e_;
        scalar mu_;

    // Used for the GPU tracking

        bool convertToFloat;

        bool validate;

        StaticHostData *staticHostData;

        TimeDepHostData *timeDepHostData;

    // Private Member Functions

        //- Disallow default bitwise copy construct
        gpuParticleCloud(const gpuParticleCloud&);

        //- Disallow default bitwise assignment
        void operator=(const gpuParticleCloud&);

        void updateCloudObject();

        //- Do a mesh search and check if the occupancy for each particle is
        //  set correctly. For debugging purposes.
        void checkParticles();

public:

    // Constructors

        //- Construct given mesh
        gpuParticleCloud(
            const fvMesh& mesh,
            const word& cloudName = "defaultCloud",
            bool readFields = true
        );

        gpuParticleCloud(
        	const fvMesh& mesh,
        	const word& cloudName,
        	IDLList<gpuParticle>& particleList
        );

    // Member Functions

        // Access

            inline const fvMesh& mesh() const;

            inline scalar rhop() const;
            inline scalar e() const;
            inline scalar mu() const;

        // Edit

            //- Move the particles under the influence of the given
            //  gravitational acceleration
            void move(const dimensionedVector& g);

            // Initialize the GPU and copy static data to it.
            void initGPU(bool validate = false);

            // Do the particle tracking on the GPU.
            void moveGPU();

        // Info

            void info() const;

        // Helper methods
        // TODO Make them private eventually

            void getCellCentres(std::vector<scalar>& Cc);

            void getNfacesPerCell(std::vector<int>& nFacesPerCell);

            void getFaceLabelsPerCell(
				std::vector<int>& faceLabelsIndex,
				std::vector<int>& faceLabels
            );

            void getFaceData(std::vector<scalar>& Cf, std::vector<scalar>& Sf);

        	void getOwnersAndNeighbours(
        		std::vector<int>& owner,
        		std::vector<int>& neighbour
        	);

        	// Fields

        	inline void flattenVectorField(
        		const volVectorField& field,
        		std::vector<scalar>& vec
        	);

        	// Particles

        	inline void getParticleData(
        		gpuParticleCloud& cloud,
        		std::vector<scalar>& pos,
        		std::vector<scalar>& Uparticle,
        		std::vector<scalar>& d,
        		std::vector<int>&occupancy
        	);

        	// Destructor
        	~gpuParticleCloud();

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "gpuParticleCloudI.H"
#include "gpuParticleCloudHelperI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
