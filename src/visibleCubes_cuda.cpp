/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <visibleCubes_cuda.h>
#include <cuda_help.h>

#include <cuda.h>

#define NUM_THREADS

namespace eqMivt
{

__global__ void cuda_updateIndex(visibleCubeGPU cubes, int size, int * cube, int * sCube, int * nocube, int * sNocube, int * cached, int * sCached, int * painted, int * sPainted)
{
	int offset = 0;
	__shared__ int id[NUM_THREADS];

	while(offset < size)
	{
		int idx = offset + threadIdx.x;
		
		if (idx < size)
		{
			// NO CUBE
			if (blockIdx.x == 0)
			{
				if (cubes[offset + idx].state == CUDA_NOCUBE)
					id[idx] = offset + idx;
				else
					id[idx] = -1;
			}
			// CUBE
			else if (blockIdx.x == 1)
			{
				if (cubes[offset + idx].state == CUDA_CUBE)
					id[idx] = offset + idx;
				else
					id[idx] = -1;
			}
			// CACHED 
			else if (blockIdx.x == 2)
			{
				if (cubes[offset + idx].state == CUDA_CACHED)
					id[idx] = offset + idx;
				else
					id[idx] = -1;
			}
			// PAINTED
			else if (blockIdx.x == 3)
			{
				if (cubes[offset + idx].state == CUDA_PAINTED)
					id[idx] = offset + idx;
				else
					id[idx] = -1;
			}
		}
		__syncthreads();
			if (blockIdx.x == 0 && threadIdx.x == 0)
			{
				for(int i=0; i<blockDim.x; i++)
				{
					if (id[i] != -1)
					{
						nocube[*sNocube] = id[i];
						(*sNocube)++;
					}
				}
			}
			// CUBE
			else if (blockIdx.x == 1 && threadIdx.x == 0)
			{
				for(int i=0; i<blockDim.x; i++)
				{
					if (id[i] != -1)
					{
						cube[*sCube] = id[i];
						(*sCube)++;
					}
				}
			}
			// CACHED 
			else if (blockIdx.x == 2 && threadIdx.x == 0)
			{
				for(int i=0; i<blockDim.x; i++)
				{
					if (id[i] != -1)
					{
						cached[*sCached] = id[i];
						(*sCached)++;
					}
				}
			}
			// PAINTED
			else if (blockIdx.x == 3 && threadIdx.x == 0)
			{
				for(int i=0; i<blockDim.x; i++)
				{
					if (id[i] != -1)
					{
						painted[*sPainted] = id[i];
						(*sPainted)++;
					}
				}
			}

		__syncthreads();

		offset += blockDim.x;
	}
}

void updateIndex(visibleCubeGPU cubes, int size, int * cube, int * sCube, int * nocube, int * sNocube, int * cached, int * sCached, int * painted, int * sPainted)
{
	cuda_updateIndex<<<4, NUM_THREADS>>>(cubes, size, cube, sCube, nocube, sNocube, cached, sCached, painted, sPainted); 	
}

}
