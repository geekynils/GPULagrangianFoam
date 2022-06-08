// Particles -------------------------------------------------------------------

inline void Foam::gpuParticleCloud::getParticleData
(
	gpuParticleCloud& cloud,
	std::vector<scalar>& pos,
	std::vector<scalar>& Uparticle,
	std::vector<scalar>& d,
	std::vector<int>&occupancy
)
{
	int nParticles = cloud.size();
	pos.resize(nParticles*4);
	Uparticle.resize(nParticles*4);
	d.resize(nParticles);
	occupancy.resize(nParticles);

	int i;

	for (
		gpuParticleCloud::iterator iter = cloud.begin();
		iter != cloud.end();
		++iter
	) {
        gpuParticle& particle = iter();
        i = particle.id();

        pos.at(i*4)     = particle.position().x();
        pos.at(i*4 + 1) = particle.position().y();
        pos.at(i*4 + 2) = particle.position().z();
        pos.at(i*4 + 3) = padVal;

        Uparticle.at(i*4) 	  = particle.U().x();
        Uparticle.at(i*4 + 1) = particle.U().y();
        Uparticle.at(i*4 + 2) = particle.U().z();
        Uparticle.at(i*4 + 3) = padVal;

        d.at(i) = particle.d();

        occupancy.at(i) = particle.cell();
    }

}

inline void Foam::gpuParticleCloud::flattenVectorField
(
	const volVectorField& field,
	std::vector<scalar>& vec
){
	int fieldLength = field.size();
	int length = fieldLength*4;

	vec.resize(length);

	for(int i=0; i<fieldLength; i++) {

		vec.at(4*i)     = field[i].x();
		vec.at(4*i + 1) = field[i].y();
		vec.at(4*i + 2) = field[i].z();
		vec.at(4*i + 3) = padVal;
	}
}



