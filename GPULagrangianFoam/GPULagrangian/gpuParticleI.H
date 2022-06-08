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

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

inline Foam::gpuParticle::trackData::trackData
(
    gpuParticleCloud& spc,
    const interpolationCellPoint<scalar>& rhoInterp,
    const interpolationCellPoint<vector>& UInterp,
    const interpolationCellPoint<scalar>& nuInterp,
    const vector& g
)
:
    spc_(spc),
    rhoInterp_(rhoInterp),
    UInterp_(UInterp),
    nuInterp_(nuInterp),
    g_(g)
{}


inline Foam::gpuParticle::gpuParticle
(
    const Cloud<gpuParticle>& c,
    const vector& position,
    const label celli,
    const scalar d,
    const vector& U
)
:
    Particle<gpuParticle>(c, position, celli),
    d_(d),
    U_(U)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

inline Foam::gpuParticleCloud& Foam::gpuParticle::trackData::spc()
{
    return spc_;
}

inline const Foam::interpolationCellPoint<Foam::scalar>&
Foam::gpuParticle::trackData::rhoInterp() const
{
    return rhoInterp_;
}

inline const Foam::interpolationCellPoint<Foam::vector>&
Foam::gpuParticle::trackData::UInterp() const
{
    return UInterp_;
}

inline const Foam::interpolationCellPoint<Foam::scalar>&
Foam::gpuParticle::trackData::nuInterp() const
{
    return nuInterp_;
}

inline const Foam::vector& Foam::gpuParticle::trackData::g() const
{
    return g_;
}


inline Foam::scalar Foam::gpuParticle::d() const
{
    return d_;
}

inline Foam::scalar Foam::gpuParticle::wallImpactDistance(const vector&) const
{
    return 0.5*d_;
}

inline Foam::vector& Foam::gpuParticle::U()
{
    return U_;
}


// ************************************************************************* //
