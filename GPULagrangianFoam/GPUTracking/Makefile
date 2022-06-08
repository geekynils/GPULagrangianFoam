libgputracking.so: GPUTracking.cu ParticleData.cu FlatMesh.cu \
                   ParticleEngine.cu  deviceManagement.cu \
                   Kernels.cu
	nvcc -I$(HOME)/NVIDIA_GPU_Computing_SDK/C/common/inc  \
         --compiler-options -Wextra,-Wall,-fPIC --shared -arch=sm_20 \
         --ptxas-options=-v \
	     --linker-options -soname,libgputracking.so $^ -o $@
clean:
	rm -rf libgputracking.so
