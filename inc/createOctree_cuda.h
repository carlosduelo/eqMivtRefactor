/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef _EQ_MIVT_CREATOR_OCTREE_CUDA_H_
#define _EQ_MIVT_CREATOR_OCTREE_CUDA_H_

#include "typedef.h"

#include <cuda_runtime.h>

namespace eqMivt
{
	void extracIsosurface(unsigned int numElements, unsigned int cubeLevel, unsigned int nLevels, float iso, index_node_t idCube, unsigned char * result, float * cube, cudaStream_t stream = 0);
}

#endif /*_EQ_MIVT_CREATOR_OCTREE_CUDA_H_*/
