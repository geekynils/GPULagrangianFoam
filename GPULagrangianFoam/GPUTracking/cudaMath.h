#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include "eclipse.h"
#include "GPUTracking.h"

// NOTE THAT vec4 is used as a three-component vector. This is because
// four components can be fetched at once from main memory, but not three.
// PTX assembly: ldu.global.v4.f32

// TODO Use C++ and call a constructor instead?
#ifdef GPU_PRECISION_DP
	#define makeVec2 make_double2
	#define makeVec3 make_double3
	#define makeVec4 make_double4
	typedef double2 vec2;
	typedef double3 vec3;
	typedef double4 vec4;
	__device__ const gpuScalar eps = 1e-9;
	__device__ const gpuScalar SMALL = 1.0e-15;
#else
	#define makeVec2 make_float2
	#define makeVec3 make_float3
	#define makeVec4 make_float4
	typedef float2 vec2;
	typedef float3 vec3;
	typedef float4 vec4;
	__device__ const gpuScalar eps = 1.19209e-07; // taken from C++ limits
	__device__ const gpuScalar SMALL = 1.0e-6;
#endif

// -----------------------------------------------------------------------------
// Overload simple operators for vec3

inline __device__ __host__ vec3 operator+(const vec3 a, const vec3 b)
{
	return makeVec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ __host__ vec3 operator-(const vec3& a, const vec3& b)
{
	return makeVec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ __host__ vec3 operator*(const vec3 a, const gpuScalar b)
{
	return makeVec3(a.x*b, a.y*b, a.z*b);
}

inline __device__ __host__ vec3 operator*(const gpuScalar a, const vec3 b)
{
	return makeVec3(a*b.x, a*b.y, a*b.z);
}

inline __device__ __host__ void operator+=(vec3& a, const vec3 b)
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __device__ __host__ vec3 operator-(const vec3& a, const gpuScalar b)
{
	return makeVec3(a.x - b, a.y - b, a.z - b);
}

// -----------------------------------------------------------------------------
// Vector functions for vec3

inline __device__ __host__ gpuScalar dotP(vec3 v1, vec3 v2)
{
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

inline __device__ __host__ vec3 normVec(vec3 v)
{
	const gpuScalar invlen = rsqrtf(v.x*v.x + v.y*v.y + v.z*v.z);

	v.x *= invlen;
	v.y *= invlen;
	v.z *= invlen;

	return makeVec3(v.x, v.y, v.z);
}

#endif
