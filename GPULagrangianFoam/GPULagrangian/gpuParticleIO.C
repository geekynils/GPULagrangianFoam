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

#include "gpuParticle.H"
#include "IOstreams.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::gpuParticle::gpuParticle
(
    const Cloud<gpuParticle>& cloud,
    Istream& is,
    bool readFields
)
:
    Particle<gpuParticle>(cloud, is, readFields)
{
    if (readFields)
    {
        if (is.format() == IOstream::ASCII)
        {
            d_ = readScalar(is);
            is >> U_;
        }
        else
        {
            is.read
            (
                reinterpret_cast<char*>(&d_),
                sizeof(d_) + sizeof(U_)
            );
        }
    }

    // Check state of Istream
    is.check("gpuParticle::gpuParticle(Istream&)");

    this->id_ = instances;
    instances++;
}


void Foam::gpuParticle::readFields(Cloud<gpuParticle>& c)
{
    if (!c.size())
    {
        return;
    }
    IOField<scalar> d(c.fieldIOobject("d", IOobject::MUST_READ));
    c.checkFieldIOobject(c, d);

    IOField<vector> U(c.fieldIOobject("U", IOobject::MUST_READ));
    c.checkFieldIOobject(c, U);

    label i = 0;
    forAllIter(Cloud<gpuParticle>, c, iter)
    {
        gpuParticle& p = iter();

        p.d_ = d[i];
        p.U_ = U[i];
        i++;
    }
}


void Foam::gpuParticle::writeFields(const Cloud<gpuParticle>& c)
{
    Particle<gpuParticle>::writeFields(c);

    label np = c.size();

    IOField<scalar> d(c.fieldIOobject("d", IOobject::NO_READ), np);
    IOField<vector> U(c.fieldIOobject("U", IOobject::NO_READ), np);

    label i = 0;
    forAllConstIter(Cloud<gpuParticle>, c, iter)
    {
        const gpuParticle& p = iter();

        d[i] = p.d_;
        U[i] = p.U_;
        i++;
    }

    d.write();
    U.write();
}


// * * * * * * * * * * * * * * * IOstream Operators  * * * * * * * * * * * * //

Foam::Ostream& Foam::operator<<(Ostream& os, const gpuParticle& p)
{
    if (os.format() == IOstream::ASCII)
    {
        os  << static_cast<const Particle<gpuParticle>&>(p)
            << token::SPACE << p.d_
            << token::SPACE << p.U_;
    }
    else
    {
        os  << static_cast<const Particle<gpuParticle>&>(p);
        os.write
        (
            reinterpret_cast<const char*>(&p.d_),
            sizeof(p.d_) + sizeof(p.U_)
        );
    }

    // Check state of Ostream
    os.check("Ostream& operator<<(Ostream&, const gpuParticle&)");

    return os;
}


// ************************************************************************* //
