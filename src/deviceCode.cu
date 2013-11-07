/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#define DEVICE_CODE

#include <typedef.h>
#include <../src/textures.cu>

namespace eqMivt
{

inline __device__ float3 _cuda_BoxToCoordinates(int3 pos, int3 realDim)
{
	float3 r;
	r.x = pos.x < -2 ? tex1Dfetch(xgrid, 0) - 1.0f : pos.x > realDim.x + 1 ? tex1Dfetch(xgrid, CUBE_INC + realDim.x+1) + 1.0f : tex1Dfetch(xgrid, CUBE_INC + pos.x);
	r.y = pos.y < -2 ? tex1Dfetch(ygrid, 0) - 1.0f : pos.y > realDim.y + 1 ? tex1Dfetch(ygrid, CUBE_INC + realDim.y+1) + 1.0f : tex1Dfetch(ygrid, CUBE_INC + pos.y);
	r.z = pos.z < -2 ? tex1Dfetch(zgrid, 0) - 1.0f : pos.z > realDim.z + 1 ? tex1Dfetch(zgrid, CUBE_INC + realDim.z+1) + 1.0f : tex1Dfetch(zgrid, CUBE_INC + pos.z);

	return r;
}

}

#include <../src/octree_cuda.cu>
#include <../src/rayCaster_cuda.cu>


