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

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::gpuParticle::move(gpuParticle::trackData& td) {

    td.switchProcessor = false;
    td.keepParticle = true;

    const polyMesh& mesh = cloud().pMesh();
    const polyBoundaryMesh& pbMesh = mesh.boundaryMesh();

    scalar deltaT = mesh.time().deltaT().value();
    scalar tEnd = (1.0 - stepFraction())*deltaT;
    scalar dtMax = tEnd;

    while (td.keepParticle && !td.switchProcessor && tEnd > SMALL) {

        if (debug) {

        	Info << "Debug tracking for particle " << id() << " " << nl
        		 << "    Pos at the beginning " << position() << " "
            	 << "(in cell " << cell() << ")" << nl
                 << "    Time = " << mesh.time().timeName()
                 << "        deltaT = " << deltaT
                 << "        tEnd = " << tEnd
                 << "        steptFraction() = " << stepFraction() << nl;
        }

        // set the lagrangian time-step
        scalar dt = min(dtMax, tEnd);

        // remember which cell the parcel is in
        // since this will change if a face is hit
        label celli = cell();

        dt *= trackToFace(position() + dt*U_, td);

        if (debug) { // Note that the position is updated by trackToFace(..)
        	Info << "    Pos at the end " << position() << nl;
        }

        tEnd -= dt;
        stepFraction() = 1.0 - tEnd/deltaT;

        // Omit the whole interpolation stuff for now and just take the velocity
        // from the current cell..
        // cellPointWeight cpw(mesh, position(), celli, face());
        // scalar rhoc = td.rhoInterp().interpolate(cpw);
        // vector Uc = td.UInterp().interpolate(cpw);
        // scalar nuc = td.nuInterp().interpolate(cpw);

        // Hack to assume constant velocity
        // scalar rhoc = 1;
        // vector Uc(1, 0, 0);
        // scalar nuc = 1;

        /*
        scalar rhop = td.spc().rhop();
        scalar magUr = mag(Uc - U_);

        scalar ReFunc = 1.0;
        scalar Re = magUr*d_/nuc;

        if (Re > 0.01)
        {
            ReFunc += 0.15*pow(Re, 0.687);
        }

        scalar Dc = (24.0*nuc/d_)*ReFunc*(3.0/4.0)*(rhoc/(d_*rhop));

        U_ = (U_ + dt*(Dc*Uc + (1.0 - rhoc/rhop)*td.g()))/(1.0 + dt*Dc);
        */

        // For now we ignore the interpolation step and take the veolcity
        // directly from the cell.

        if(celli != cell()) {
        	U_ = U_ + 0.5 * mesh.lookupObject<const volVectorField>("U")[celli];
        }

        if (onBoundary() && td.keepParticle)
        {
            // Bug fix.  HJ, 25/Aug/2010
            if (face() > -1)
            {
                if (isType<processorPolyPatch>(pbMesh[patch(face())]))
                {
                    td.switchProcessor = true;
                }
            }
        }
    }

    if (debug)
    {
    	Info << "Done moving particle, going to the next time step." << nl;
    }

    return td.keepParticle;
}


bool Foam::gpuParticle::hitPatch
(
    const polyPatch&,
    gpuParticle::trackData&,
    const label
)
{
    return false;
}


bool Foam::gpuParticle::hitPatch
(
    const polyPatch&,
    int&,
    const label
)
{
    return false;
}


void Foam::gpuParticle::hitProcessorPatch
(
    const processorPolyPatch&,
    gpuParticle::trackData& td
)
{
    td.switchProcessor = true;
}


void Foam::gpuParticle::hitProcessorPatch
(
    const processorPolyPatch&,
    int&
)
{}


void Foam::gpuParticle::hitWallPatch
(
    const wallPolyPatch& wpp,
    gpuParticle::trackData& td
)
{
/* Just invert the direction... HELLO PHYSICS!?
 *
    vector nw = wpp.faceAreas()[wpp.whichFace(face())];
    nw /= mag(nw);

    scalar Un = U_ & nw;
    vector Ut = U_ - Un*nw;

    if (Un > 0)
    {
        U_ -= (1.0 + td.spc().e())*Un*nw;
    }

    U_ -= td.spc().mu()*Ut;
*/

	// Use a simple specular reflection instead.

	vector n = wpp.faceAreas()[wpp.whichFace(face())];
	n /= mag(n);

	U_ -= 2 * (U_ & n) * n;
}


void Foam::gpuParticle::hitWallPatch
(
    const wallPolyPatch&,
    int&
)
{}


void Foam::gpuParticle::hitPatch
(
    const polyPatch&,
    gpuParticle::trackData& td
)
{
    td.keepParticle = false;
}


void Foam::gpuParticle::hitPatch
(
    const polyPatch&,
    int&
)
{}


void Foam::gpuParticle::transformProperties(const tensor& T)
{
    Particle<gpuParticle>::transformProperties(T);
    U_ = transform(T, U_);
}


void Foam::gpuParticle::transformProperties(const vector& separation)
{
    Particle<gpuParticle>::transformProperties(separation);
}


// ************************************************************************* //
