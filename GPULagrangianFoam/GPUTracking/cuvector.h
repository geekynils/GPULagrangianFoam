#ifndef CUVECTOR_H
#define CUVECTOR_H

#include <vector>
#include <cstdio>

#include <cuda.h>

/**
 * Wrapper around std::vector. Holds a reference to the actual vector and the
 * concerning device pointer. Memory on the device is automatically allocated
 * upon construction (using cudaMalloc(..)) and freed when the object is
 *  destroyed (using cudaFree(..)).
 */
template<class T>
class cuvector {

	//!
	//! Pointer to device memory.
	T* devicePtr_;

	//!
	//! Reference to the host vector.
	std::vector<T>& hostVector_;

	//!
	//! Disallow copying.
	cuvector& operator=(const cuvector&);
	cuvector(const cuvector &);

	//!
	//! Wraps the CUDA API for error checking.
	void checkCUDAError(const char *msg) {

		cudaError_t err = cudaGetLastError();
		if(cudaSuccess != err) {
			fprintf(stderr, "Cuda error in cuvector: %s: %s.\n", msg,
				cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}


public:
	
	// TODO error checking
	//!
	//! Constructs cuvector from a given std::vector. Memory on the device is
	//! allocated.
	inline cuvector(std::vector<T>& hostVector)
		: hostVector_(hostVector) {

		cudaMalloc(
			(void**)&devicePtr_,
			sizeof(T) * hostVector_.size()
		);

		checkCUDAError("When trying to construct cuvector");
	}

	//!
	//! Frees memory on the device to which the device pointer was pointing.
	inline ~cuvector() {

		cudaFree(devicePtr_);

		checkCUDAError("When trying to destroy cuvector");
	}


	//!
	//! Terminology is from the viewpoint of the host: Upload means that data
	//! from the host vector is copied to the device memory.
	inline cudaError_t upload() const {

		return cudaMemcpy(
				devicePtr_,
				&(hostVector_[0]),
				sizeof(T) * hostVector_.size(),
			cudaMemcpyHostToDevice);
	}
	

	//!
	//! Copies memory from the device to the host vector.
	inline cudaError_t download() const {

		return cudaMemcpy(
				&(hostVector_[0]),
				devicePtr_,
				sizeof(T) * hostVector_.size(),
			cudaMemcpyDeviceToHost);
	}
	
	//! Access elements without bounds checking. Wraps the concerning function
	//! for std::vector.
	inline const T& operator[] (unsigned int i) const {
		return hostVector_[i];
	}

	inline T& operator[] (unsigned int i) {
		return hostVector_[i];
	}

	//! Access elements with bounds checking. Wraps the concerning function for
	//! std::vector.
	inline T& at (unsigned int i) {
		return hostVector_.at(i);
	}

	inline const T& at (unsigned int i) const {
		return hostVector_.at(i);
	}

	//!
	//! Returns the number of elements in the vector.
	inline unsigned int size() const {
		return hostVector_.size();
	}

	//!
	//! Returns the size of the data it is holding in bytes.
	inline long dataSize() const {
		return hostVector_.size() * sizeof(T);
	}


	//!
	//! Returns a pointer to the device memory.
	inline T* devicePtr() {
		return devicePtr_;
	}
};

#endif
