/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef _EQ_MIVT_CREATOR_OCTREE_CUDA_H_
#define _EQ_MIVT_CREATOR_OCTREE_CUDA_H_

#include "typedef.h"

namespace eqMivt
{
	void extracIsosurface(unsigned int numElements, unsigned int cubeLevel, unsigned int nLevels, float iso, index_node_t idCube, unsigned int * result, float * cube);
}

#endif /*_EQ_MIVT_CREATOR_OCTREE_CUDA_H_*/
