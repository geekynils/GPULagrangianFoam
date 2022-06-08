inline vec3 ParticleEngine::updateVelocity(
	const vec3& Uparticle,
	const vec3& Ufield) const {

	return 0.5 * Uparticle + 0.5 * Ufield;
}
