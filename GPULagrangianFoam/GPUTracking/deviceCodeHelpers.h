inline __device__ vec3 getVec3(vec4 *data, int i) {
	return makeVec3(data[i].x, data[i].y, data[i].z);
}