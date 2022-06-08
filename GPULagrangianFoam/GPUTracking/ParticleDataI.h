// Getters

inline vec3 ParticleData::getParticlePosition(const int& particleLabel) const {
	return getVec3(particlePositions, particleLabel);
}

inline vec3 ParticleData::getEstimatedEndPosition(const int& particleLabel) const {
	return getVec3(estimatedEndPositions, particleLabel);
}

inline vec3 ParticleData::getU(const int& particleLabel) const {
	return getVec3(U, particleLabel);
}

inline vec3 ParticleData::getParticleDiameter(const int& particleLabel) const {
	return getVec3(diameters, particleLabel);
}

// Setters

inline void ParticleData::setParticlePosition(
	const vec3& position, const int& particleLabel) {
	setVector(particlePositions, position, particleLabel);
}

inline void ParticleData::setEstimatedEndPosition(
	const vec3& endPos,  const int& particleLabel) {
	setVector(estimatedEndPositions, endPos, particleLabel);
}

inline void ParticleData::setU(const vec3& uIn, const int& particleLabel) {
	setVector(U, uIn, particleLabel);
}

inline void ParticleData::setParticleDiameter(const vec3& d,
	const int& particleLabel) {
	setVector(diameters, d, particleLabel);
}
