/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_VISIBLE_CUBES_CUDA_H
#define EQ_MIVT_VISIBLE_CUBES_CUDA_H

#include <visibleCubes.h>

namespace eqMivt
{
void updateIndex(visibleCubeGPU cubes, int size, int * cube, int * sCube, int * nocube, int * sNocube, int * cached, int * sCached, int * painted, int * sPainted);
}

#endif /*EQ_MIVT_VISIBLE_CUBES_CUDA_H*/
