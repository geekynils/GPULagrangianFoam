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

Application
	gpuLagrangianFoam

Description
	Takes a static vector field as input and tracks particles through a given
	mesh.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"

#include "gpuParticleCloud.H"

// Avoids linker errors
label gpuParticle::instances;

int main(int argc, char *argv[])
{
	// Parse arguments ---------------------------------------------------------

    argList::noParallel();

    //- Execute tracking on the GPU
    argList::validOptions.insert("gpu", "");

    //- Validate the tracking results using an octree to search the particles in
    // the mesh and validate occpancy information after each time step,
    // obviously slow.
    argList::validOptions.insert("validate", "");

    Foam::argList args(argc, argv);

    if (!args.checkRootCase()) {
        Foam::FatalError.exit();
    }

    // -------------------------------------------------------------------------

	#include "createTime.H"
	#include "createMesh.H"
	#include "createFields.H"

    bool gpu = args.optionFound("gpu");
    bool validate = args.optionFound("validate");

    if(gpu)
    	cloud.initGPU(validate);

	Info << "\nStarting time loop \n" << endl;

	while (runTime.loop())
	{
		Info << "Time = " << runTime.timeName() << nl << endl;
		Info << "Evolving cloud.." << endl;

		if(gpu) {
			cloud.moveGPU();
		} else {
			cloud.info();
			cloud.move(g);
		}

		runTime.write();

		Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
			 << "  ClockTime = " << runTime.elapsedClockTime() << " s"
			 << nl << nl;
	}

	Info << "End\n" << endl;

    return 0;
}
