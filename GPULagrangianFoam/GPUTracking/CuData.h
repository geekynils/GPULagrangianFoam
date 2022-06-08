#ifndef CU_DATA_H
#define CU_DATA_H

#include <cuda.h>
#include "cudaMath.h"

class CuData
{

protected:

	//!
	//! Gets a vector at the index of a label.
	inline vec3 getVec3(
		const cuvector<gpuScalar>& dataVector,
		const int& label
	) const {

		return makeVec3(
			dataVector.at(label*4),
			dataVector.at(label*4 + 1),
			dataVector.at(label*4 + 2)
		);
	}
	
	//!
	//! Sets dataVector at the concerning position to the values of vec.
	inline void setVector(
		cuvector<gpuScalar>& dataVector,
		const vec3& vec,
		const int& label
	) {
		dataVector.at(label*4) = vec.x;
		dataVector.at(label*4 + 1) = vec.y;
		dataVector.at(label*4 + 2) = vec.z;
	}
	
public:
	
	//!
	//! Return the size of the object in bytes.
	virtual long getDataSize() = 0;

};

#endif
