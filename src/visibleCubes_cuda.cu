/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <typedef.h>
#include <visibleCubes_cuda.h>
#include <cuda_help.h>

#include <cuda.h>

#include <cstdio>
#include <iostream>

namespace eqMivt
{

__global__ void cuda_updateIndex(visibleCubeGPU cubes, int size, int * cube, int * sCube, int * nocube, int * sNocube, int * cached, int * sCached, int * painted, int * sPainted)
{
	int idx = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;

	if (idx < size)
	{
		// NO CUBE
		if (cubes[idx].state == CUDA_NOCUBE)
			nocube[atomicAdd(sNocube, 1)] = idx;
		// CUBE
		else if (cubes[idx].state == CUDA_CUBE)
			cube[atomicAdd(sCube, 1)] = idx;
		// CACHED 
		else if (cubes[idx].state == CUDA_CACHED)
			cached[atomicAdd(sCached, 1)] = idx;
		// PAINTED
		else if (cubes[idx].state == CUDA_PAINTED)
			painted[atomicAdd(sPainted, 1)] = idx;
	}
}

void updateIndex(visibleCubeGPU cubes, int size, int * cube, int * sCube, int * nocube, int * sNocube, int * cached, int * sCached, int * painted, int * sPainted)
{
	dim3 threads = eqMivt::getThreads(size);
	dim3 blocks = eqMivt::getBlocks(size);

	cuda_updateIndex<<<blocks, threads>>>(cubes, size, cube, sCube, nocube, sNocube, cached, sCached, painted, sPainted);

	#ifndef NDEBUG 
	if (cudaSuccess != cudaDeviceSynchronize())
	{
		std::cout<<"Launch kernel: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	#endif
}

}
