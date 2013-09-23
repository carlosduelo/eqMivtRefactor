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

#define NUM_THREADS 256

namespace eqMivt
{

__global__ void cuda_updateIndex(visibleCubeGPU cubes, int size, int * cube, int * sCube, int * nocube, int * sNocube, int * cached, int * sCached, int * painted, int * sPainted)
{
	int idx = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;

	if (idx < size)
	{
		// NO CUBE
		if (cubes[idx].state == CUDA_NOCUBE)
			cube[atomicAdd(sCube, 1)] = idx;
		// CUBE
		else if (cubes[idx].state == CUDA_CUBE)
			nocube[atomicAdd(sNocube, 1)] = idx;
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
	cudaDeviceSynchronize();
	std::cout<<"Launch kernel: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
	#endif
}

}

#if 0
int main(int argc, char ** argv)
{
	int start = 2000;//*2000;
	int dim = 4*start;

	eqMivt::VisibleCubes		vc;
	vc.reSize(dim);
	vc.init();

	std::vector<int> c = vc.getListCubes(CUDA_CUBE);
	std::vector<int> nc = vc.getListCubes(CUDA_NOCUBE);
	std::vector<int> ca = vc.getListCubes(CUDA_CACHED);
	std::vector<int> p = vc.getListCubes(CUDA_PAINTED);
	std::cout<<"CUBE size "<<c.size()<<std::endl;
	std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
	std::cout<<"CACHED size "<<ca.size()<<std::endl;
	std::cout<<"PAINTED size "<<p.size()<<std::endl;

	nc = vc.getListCubes(CUDA_NOCUBE);
	for(std::vector<int>::iterator it = nc.begin(); it!= nc.end(); ++it)
	{
		eqMivt::visibleCube_t * s = vc.getCube(*it);
		s->pixel = *it;
		s->cubeID = 21;

		if ((*it) % 4 == 0)
			s->state = CUDA_NOCUBE;
		if ((*it) % 4 == 1)
			s->state = CUDA_CUBE;
		if ((*it) % 4 == 2)
			s->state = CUDA_CACHED;
		if ((*it) % 4 == 3)
			s->state = CUDA_PAINTED;
	}

	c = vc.getListCubes(CUDA_CUBE);
	nc = vc.getListCubes(CUDA_NOCUBE);
	ca = vc.getListCubes(CUDA_CACHED);
	p = vc.getListCubes(CUDA_PAINTED);
	std::cout<<"CUBE size "<<c.size()<<std::endl;
	std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
	std::cout<<"CACHED size "<<ca.size()<<std::endl;
	std::cout<<"PAINTED size "<<p.size()<<std::endl;


	int * cubeC		= new int[1+dim];
	int * nocubeC	= new int[1+dim];
	int * cachedC	= new int[1+dim];
	int * paintedC	= new int[1+dim];
	int * cubeG		= 0;
	int * nocubeG	= 0;
	int * cachedG	= 0;
	int * paintedG	= 0;
	if ( cudaSuccess != cudaMalloc((void**)&cubeG, (1+dim)*sizeof(int)) ||
		 cudaSuccess != cudaMalloc((void**)&nocubeG, (1+dim)*sizeof(int)) ||
		 cudaSuccess != cudaMalloc((void**)&cachedG, (1+dim)*sizeof(int)) ||
		 cudaSuccess != cudaMalloc((void**)&paintedG, (1+dim)*sizeof(int)) ||
		 cudaSuccess != cudaMemset((void*)cubeG, 0, (1+dim)*sizeof(int)) ||
		 cudaSuccess != cudaMemset((void*)nocubeG, 0, (1+dim)*sizeof(int)) ||
		 cudaSuccess != cudaMemset((void*)cachedG, 0, (1+dim)*sizeof(int)) ||
		 cudaSuccess != cudaMemset((void*)paintedG, 0, (1+dim)*sizeof(int)) 
		)
	{
		std::cerr<<"No allocating: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return 0;
	}

	std::cout<<"Launch kernel to process "<<vc.getSizeGPU()<<" elements "<<std::endl;

	updateIndex(vc.getVisibleCubesGPU(), vc.getSizeGPU(), cubeG+1, cubeG, nocubeG+1, nocubeG, cachedG+1, cachedG, paintedG+1, paintedG);

	if (	cudaSuccess != cudaMemcpy((void*)cubeC, (void*)cubeG, (1+dim)*sizeof(int), cudaMemcpyDeviceToHost) ||
			cudaSuccess != cudaMemcpy((void*)nocubeC, (void*)nocubeG, (1+dim)*sizeof(int), cudaMemcpyDeviceToHost) ||
			cudaSuccess != cudaMemcpy((void*)cachedC, (void*)cachedG, (1+dim)*sizeof(int), cudaMemcpyDeviceToHost) ||
			cudaSuccess != cudaMemcpy((void*)paintedC, (void*)paintedG, (1+dim)*sizeof(int), cudaMemcpyDeviceToHost)
		)
	{
		std::cerr<<"No copy: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return 0;
	}

	std::cout<<"CUBE "<<cubeC[0]<<std::endl;
	//for(int i=1; i<=25; i++)
	//std::cout<<cubeC[i]<<std::endl;
	
	std::cout<<"NO CUBE "<<nocubeC[0]<<std::endl;
	//for(int i=1; i<=25; i++)
	//std::cout<<nocubeC[i]<<std::endl;
	
	std::cout<<"CACHED "<<cachedC[0]<<std::endl;
	//for(int i=1; i<=25; i++)
	//std::cout<<cachedC[i]<<std::endl;
	
	std::cout<<"PAINTED "<<paintedC[0]<<std::endl;
	//for(int i=1; i<=25; i++)
	//std::cout<<paintedC[i]<<std::endl;

	delete[] cubeC;
	delete[] nocubeC;
	delete[] cachedC;
	delete[] paintedC;
	cudaFree(cubeC);
	cudaFree(nocubeC);
	cudaFree(cachedC);
	cudaFree(paintedC);

	vc.destroy();

	std::cout<<"Test OK!"<<std::endl;
}
#endif
