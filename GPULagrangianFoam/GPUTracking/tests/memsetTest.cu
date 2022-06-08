#include <cstdio>
#include <vector>

#include "cuvector.h"
#include "deviceManagement.h"

__global__ void someKernel(int* a)
{
	int i = threadIdx.x;

	a[i] = i;
}

int main()
{
	std::vector<int> a(10);
	
	cuvector<int> cua(a);
	
	cua.allocateDevice();

	cuMemsetIntegral(cua, 3);
	
	someKernel<<<1,10>>>(cua.devicePtr());

	cua.download();

	for(int i=0; i<10; i++)
		printf("%i ", cua.at(i));

	printf("\n");

	return 0;
}
