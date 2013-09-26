/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef _EQ_MIVT_CREATOR_OCTREE_CUDA_H_
#define _EQ_MIVT_CREATOR_OCTREE_CUDA_H_

#include "typedef.h"

	size_t	octreeConstructorGetFreeMemory();	
	eqMivt::index_node_t * octreeConstructorCreateResult(int size);
	bool	octreeConstructorCopyResult(eqMivt::index_node_t * cpuResult, eqMivt::index_node_t * gpuResult, int size);
	void	octreeConstructorDestroyResult(eqMivt::index_node_t * result);
	void	octreeConstructorComputeCube(eqMivt::index_node_t * cubes, int size, eqMivt::index_node_t startID, float iso, float * cube, int nodeLevel, int nLevels, int dimNode, int cubeDim, int coorCubeStart[3]);

#endif /*_EQ_MIVT_CREATOR_OCTREE_CUDA_H_*/
