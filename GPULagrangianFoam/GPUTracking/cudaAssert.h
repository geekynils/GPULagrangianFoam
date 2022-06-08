#ifndef CUDA_ASSERT_H
#define CUDA_ASSERT_H

#ifdef DEBUG

#include "eclipse.h"

// This flag is read at the beginning of every kernel if it's not true the
// kernel returns. This way no more kernels are launched once an assertion
// fails.
__device__ bool run=true;

// Trap cannot be used here, because if the kernel is interrupted with trap the
// output of printf is never transferred to the host and never printed out.
#define cudaAssert(condition) \
if (!(condition)){ printf("Assertion %s failed!\n", #condition); run=false; }

#define checkRun if(!run) asm("trap;");

#define info(x, ...) printf(x, __VA_ARGS__)

#else

#define cudaAssert(condition)
#define checkRun
#define info(x, ...)

#endif

#endif
