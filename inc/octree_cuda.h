/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef _EQ_MIVT_OCTREE_CUDA_H_
#define _EQ_MIVT_OCTREE_CUDA_H_

#include <typedef.h>

#include <cuda_runtime.h>

namespace eqMivt
{
	void getBoxIntersectedOctree(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float3 LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int finalLevel, int numElements, visibleCubeGPU_t visibleGPU, indexVisibleCubeGPU_t  indexVisibleCubesGPU, int3 realDim, cudaStream_t stream);

void insertOctreePointers(index_node_t ** octreeGPU, int * sizes, index_node_t * memoryGPU, int levels);
}


#endif
