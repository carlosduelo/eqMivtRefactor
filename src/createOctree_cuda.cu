/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#define BLOCK_SIZE 128

#include "createOctree_cuda.h"
#include "mortonCodeUtil.h"
#include "cuda_help.h"
#include <cutil_math.h>

#include "cuda_runtime.h"

#include <iostream>

#define posToIndex(i,j,k,d) ((k)+(j)*(d)+(i)*(d)*(d))

namespace eqMivt
{

__device__ bool cuda_checkIsosurface(int x, int y, int z, int dim, float * cube, float isosurface)
{
	bool sign = (cube[posToIndex(x, y, z, dim)] - isosurface) < 0;

	if (((cube[posToIndex(x, y, z+1, dim)] - isosurface) < 0) != sign)
		return true;
	if (((cube[posToIndex(x, y+1, z, dim)] - isosurface) < 0) != sign)
		return true;
	if (((cube[posToIndex(x, y+1, z+1, dim)] - isosurface) < 0) != sign)
		return true;
	if (((cube[posToIndex(x+1, y, z, dim)] - isosurface) < 0) != sign)
		return true;
	if (((cube[posToIndex(x+1, y, z+1, dim)] - isosurface) < 0) != sign)
		return true;
	if (((cube[posToIndex(x+1, y+1, z, dim)] - isosurface) < 0) != sign)
		return true;
	if (((cube[posToIndex(x+1, y+1, z+1, dim)] - isosurface) < 0) != sign)
		return true;

	return false;
}

__global__ void cuda_extracIsosurface(unsigned int numElements, unsigned int cubeLevel, unsigned int nLevels, float iso, index_node_t idCube, unsigned int * result, float * cube)
{
	unsigned int tid = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < numElements)
	{
		index_node_t id = idCube + tid;
		int3 coord = getMinBoxIndex2(id, nLevels, nLevels);
		int cubeDim = (nLevels - cubeLevel) * (nLevels - cubeLevel) + 2 * CUBE_INC;
		if ( cuda_checkIsosurface(coord.x, coord.y, coord.z, cubeDim, cube, iso))
		{
			unsigned int mask = 1 << (tid % 32);
			atomicOr(&result[tid/32], mask);
		}
	}
}

void extracIsosurface(unsigned int numElements, unsigned int cubeLevel, unsigned int nLevels, float iso, index_node_t idCube, unsigned int * result, float * cube)
{
	dim3 threads = getThreads(numElements);
	dim3 blocks = getBlocks(numElements);
	cuda_extracIsosurface<<<blocks, threads>>>(numElements, cubeLevel, nLevels, iso, idCube, result, cube);
}

}
