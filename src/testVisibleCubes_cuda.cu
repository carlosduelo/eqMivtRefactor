/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <testVisibleCubes_CUDA.h>
#include <cuda_help.h>

#include <cuda.h>

__global__ void cuda_updateCubesGPU(eqMivt::visibleCubeGPU cubes, int size, eqMivt::statusCube status)
{
	int idx = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;
	
	if (idx < size)
	{
		cubes[idx].state = status;
	}

	return;
}

void test_updateCubesGPU(eqMivt::visibleCubeGPU cubes, int size, eqMivt::statusCube status)
{
	dim3 threads = eqMivt::getThreads(size);
	dim3 blocks = eqMivt::getBlocks(size);

	cuda_updateCubesGPU<<<blocks,threads, 0, 0>>>(cubes, size, status);
}
