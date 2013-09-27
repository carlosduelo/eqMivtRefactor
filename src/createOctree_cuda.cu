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

	size_t	octreeConstructorGetFreeMemory()
	{
		size_t total = 0;
		size_t free = 0;

		if (cudaSuccess != cudaMemGetInfo(&free, &total))
		{
			std::cerr<<"LRUCache: Error getting memory info"<<std::endl;
			return false;
		}
		
		return free;
	}


	eqMivt::index_node_t * octreeConstructorCreateResult(int size)
	{
		eqMivt::index_node_t * result  = 0;
		size_t sizeR = (size)*sizeof(eqMivt::index_node_t); 

		if (cudaSuccess != (cudaMalloc((void **)&result, sizeR)))
		{
			return 0;
		}

		return result;
	}

	bool	octreeConstructorCopyResult(eqMivt::index_node_t * cpuResult, eqMivt::index_node_t * gpuResult, int size)
	{
		size_t sizeR = (size)*sizeof(eqMivt::index_node_t); 

		if (cudaSuccess != (cudaMemcpy((void*)cpuResult, (void*)gpuResult, sizeR, cudaMemcpyDeviceToHost)))
		{
			return false;
		}
		return true;
	}

	void	octreeConstructorDestroyResult(eqMivt::index_node_t * result)
	{
		if (result != 0)
			cudaFree(result);
	}

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

	__global__ void	cuda_octreeConstructorComputeCube(eqMivt::index_node_t * cubes, int size, eqMivt::index_node_t startID, float iso, float * cube, int nodeLevel, int nLevels, int dimNode, int cubeDim, int3 coorCubeStart)
	{
		unsigned int tid = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

		if (tid < size)
		{
			eqMivt::index_node_t id = startID + tid;

			int3 coorNodeStart = eqMivt::getMinBoxIndex2(id, nodeLevel, nLevels);
			int3 coorNodeFinish= coorNodeStart + dimNode - 1;
			coorNodeStart = coorNodeStart-coorCubeStart;
			coorNodeFinish = coorNodeFinish-coorCubeStart;

			for(int x=coorNodeStart.x; x<=coorNodeFinish.x; x++)
			{
				for(int y=coorNodeStart.y; y<=coorNodeFinish.y; y++)
				{
					for(int z=coorNodeStart.z; z<=coorNodeFinish.z; z++)
					{	
						if ( cuda_checkIsosurface(x, y, z, cubeDim, cube, iso))
						{
							cubes[tid] = id;
							return;
						}
					}
				}
			}

			cubes[tid] = 0;
			return;
		}
	}

	void	octreeConstructorComputeCube(eqMivt::index_node_t * cubes, int size, eqMivt::index_node_t startID, float iso, float * cube, int nodeLevel, int nLevels, int dimNode, int cubeDim, int  coorCubeStart[3])
	{
		dim3 threads = eqMivt::getThreads(size);
		dim3 blocks = eqMivt::getBlocks(size);
		cuda_octreeConstructorComputeCube<<<blocks, threads>>>(cubes, size, startID, iso, cube, nodeLevel, nLevels, dimNode, cubeDim, make_int3(coorCubeStart[0], coorCubeStart[1], coorCubeStart[2]));
		return;
	}
