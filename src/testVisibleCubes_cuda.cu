/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <testVisibleCubes_CUDA.h>
#include <cuda_help.h>
#include <typedef.h>

#include <cuda.h>

__global__ void cuda_updateCubesGPU(eqMivt::visibleCubeGPU cubes, int size)
{
	int idx = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;
	
	if (idx < size)
	{
		cubes[idx].state = CUDA_PAINTED;
	}

	return;
}

void test_updateCubesGPU(eqMivt::visibleCubeGPU cubes, int size)
{
	dim3 threads = eqMivt::getThreads(size);
	dim3 blocks = eqMivt::getBlocks(size);

	cuda_updateCubesGPU<<<blocks,threads, 0, 0>>>(cubes, size);
}
