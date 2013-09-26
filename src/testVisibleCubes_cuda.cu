/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <testVisibleCubes_cuda.h>
#include <cuda_help.h>
#include <iostream>

#include <cuda.h>

__global__ void cuda_updateCubesGPU(eqMivt::visibleCubeGPU cubes, eqMivt::indexVisibleCubeGPU index, int sizeG, int sizeC, eqMivt::statusCube status)
{
	int idx = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;
	if (idx < sizeG)
	{
		idx = index[idx];
		
		if (idx < sizeC)
		{
			cubes[idx].state = status;
		}
	}
	return;
}

void test_updateCubesGPU(eqMivt::visibleCubeGPU cubes, eqMivt::indexVisibleCubeGPU index, int sizeG, int sizeC, eqMivt::statusCube status)
{
	dim3 threads = eqMivt::getThreads(sizeG);
	dim3 blocks = eqMivt::getBlocks(sizeG);

	cuda_updateCubesGPU<<<blocks,threads, 0, 0>>>(cubes, index, sizeG, sizeC, status);
	#ifndef NDEBUG 
	if (cudaSuccess != cudaDeviceSynchronize())
	{
		std::cout<<"Launch kernel: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	#endif
}

__global__ void cuda_randomNOCUBE_To_NOCUBEorCUBE(eqMivt::visibleCubeGPU cubes, eqMivt::indexVisibleCubeGPU index, int sizeG, int sizeC, eqMivt::index_node_t idS, eqMivt::index_node_t idE)
{
	int idx = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;
	if (idx < sizeG)
	{
		idx = index[idx];
		
		if (idx < sizeC)
		{
			cubes[idx].state = idx % 2 == 1 ? CUDA_CUBE : CUDA_NOCUBE;
			cubes[idx].id = idS + (idE-idS)/idx;
		}
	}
	return;
}
 
void test_randomNOCUBE_To_NOCUBEorCUBE(eqMivt::visibleCubeGPU cubes, eqMivt::indexVisibleCubeGPU index, int sizeG, int sizeC, eqMivt::index_node_t idS, eqMivt::index_node_t idE)
{
	dim3 threads = eqMivt::getThreads(sizeG);
	dim3 blocks = eqMivt::getBlocks(sizeG);

	cuda_randomNOCUBE_To_NOCUBEorCUBE<<<blocks,threads, 0, 0>>>(cubes, index, sizeG, sizeC, idS, idE);
	#ifndef NDEBUG 
	if (cudaSuccess != cudaDeviceSynchronize())
	{
		std::cout<<"Launch kernel: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	#endif
}
