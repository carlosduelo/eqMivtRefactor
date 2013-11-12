/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "octree_cuda.h"

#ifndef DEVICE_CODE
#include <../src/textures.cu>
#endif

#include "cuda_help.h"
#include "mortonCodeUtil.h"

#include "cutil_math.h"

#include <iostream>
#include <fstream>

namespace eqMivt
{
#ifndef DEVICE_CODE
inline __device__ float3 _cuda_BoxToCoordinates(int3 pos, int3 realDim)
{
	float3 r;
	r.x = pos.x >= realDim.x ? tex1Dfetch(xgrid, CUBE_INC + realDim.x-1) : tex1Dfetch(xgrid, CUBE_INC + pos.x);
	r.y = pos.y >= realDim.y ? tex1Dfetch(ygrid, CUBE_INC + realDim.y-1) : tex1Dfetch(ygrid, CUBE_INC + pos.y);
	r.z = pos.z >= realDim.z ? tex1Dfetch(zgrid, CUBE_INC + realDim.z-1) : tex1Dfetch(zgrid, CUBE_INC + pos.z);

	return r;
}
#endif
/*
 **********************************************************************************************
 ****** GPU Octree functions ******************************************************************
 **********************************************************************************************
 */

__device__ inline bool _cuda_checkRangeGrid(index_node_t * elements, index_node_t index, int min, int max)
{
	return  index == elements[min] 	|| 
		index == elements[max]	||
		(elements[min] < index && elements[max] > index);
}

__device__ int _cuda_binary_search_closer_Grid(index_node_t * elements, index_node_t index, int min, int max)
{
#if 1
	bool end = false;
	bool found = false;
	int middle = 0;

	while(!end && !found)
	{
		int diff 	= max-min;
		middle	= min + (diff / 2);
		if (middle % 2 == 1) middle--;

		end 		= diff <= 1;
		found 		=  _cuda_checkRangeGrid(elements, index, middle, middle+1);
		if (index < elements[middle])
			max = middle-1;
		else //(index > elements[middle+1])
			min = middle + 2;
	}
	return middle;
#endif
#if 0
	while(1)
	{
		int diff = max-min;
		unsigned int middle = min + (diff / 2);
		if (diff <= 1)
		{
			if (middle % 2 == 1) middle--;
			return middle;
		}
		else
		{
			if (middle % 2 == 1) middle--;

			if (_cuda_checkRangeGrid(elements, index, middle, middle+1))
				return middle;
			else if (index < elements[middle])
			{
				max = middle-1;
			}
			else if (index > elements[middle+1])
			{
				min = middle + 2;
			}
			#if 0
			// XXX en cuda me arriesgo... malo...
			else
				std::cout<<"Errro"<<std::endl;
			#endif
		}
	}
#endif
}

__device__  bool _cuda_searchSecuentialGrid(index_node_t * elements, index_node_t index, int min, int max)
{
	bool find = false;
	for(int i=min; i<max; i+=2)
		if (_cuda_checkRangeGrid(elements, index, i, i+1))
			find = true;

	return find;
}

__device__ bool _cuda_RayAABB(index_node_t index, float3 origin, float3 dir,  float * tnear, float * tfar, int nLevels, int3 realDim)
{
	int3 minBoxC;
	int3 maxBoxC;
	int level;
	minBoxC = getMinBoxIndex(index, &level, nLevels); 
	if (minBoxC.x >= realDim.x || minBoxC.y >= realDim.y || minBoxC.y >= realDim.y)
		return false;
	int dim = (1<<(nLevels-level));
	maxBoxC.x = dim + minBoxC.x;
	maxBoxC.y = dim + minBoxC.y;
	maxBoxC.z = dim + minBoxC.z;
	float3 minBox = _cuda_BoxToCoordinates(minBoxC, realDim);
	float3 maxBox = _cuda_BoxToCoordinates(maxBoxC, realDim);

	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	float divx = 1.0f / dir.x;
	if (divx >= 0.0f)
	{
		tmin = (minBox.x - origin.x)*divx;
		tmax = (maxBox.x - origin.x)*divx;
	}
	else
	{
		tmin = (maxBox.x - origin.x)*divx;
		tmax = (minBox.x - origin.x)*divx;
	}
	float divy = 1.0f / dir.y;
	if (divy >= 0.0f)
	{
		tymin = (minBox.y - origin.y)*divy;
		tymax = (maxBox.y - origin.y)*divy;
	}
	else
	{
		tymin = (maxBox.y - origin.y)*divy;
		tymax = (minBox.y - origin.y)*divy;
	}

	if ( (tmin > tymax) || (tymin > tmax) )
		return false;

	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	float divz = 1.0f / dir.z;
	if (divz >= 0.0f)
	{
		tzmin = (minBox.z - origin.z)*divz;
		tzmax = (maxBox.z - origin.z)*divz;
	}
	else
	{
		tzmin = (maxBox.z - origin.z)*divz;
		tzmax = (minBox.z - origin.z)*divz;
	}

	if ( (tmin > tzmax) || (tzmin > tmax) )
		return false;

	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if (tmin<0.0f)
	 	*tnear=0.0f;
	else
		*tnear=tmin;
	*tfar=tmax;

	return *tnear < *tfar;
}

__device__ bool _cuda_RayAABB2(float3 origin, float3 dir,  float * tnear, float * tfar, int nLevels, int3 minBoxC, int level, int3 realDim)
{
	if (minBoxC.x >= realDim.x || minBoxC.y >= realDim.y || minBoxC.y >= realDim.y)
		return false;

	int3 maxBoxC;
	int dim = (1<<(nLevels-level));
	maxBoxC.x = dim + minBoxC.x;
	maxBoxC.y = dim + minBoxC.y;
	maxBoxC.z = dim + minBoxC.z;
	float3 minBox = _cuda_BoxToCoordinates(minBoxC, realDim);
	float3 maxBox = _cuda_BoxToCoordinates(maxBoxC, realDim);

	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	float divx = 1.0f / dir.x;
	if (divx >= 0.0f)
	{
		tmin = (minBox.x - origin.x)*divx;
		tmax = (maxBox.x - origin.x)*divx;
	}
	else
	{
		tmin = (maxBox.x - origin.x)*divx;
		tmax = (minBox.x - origin.x)*divx;
	}
	float divy = 1.0f / dir.y;
	if (divy >= 0.0f)
	{
		tymin = (minBox.y - origin.y)*divy;
		tymax = (maxBox.y - origin.y)*divy;
	}
	else
	{
		tymin = (maxBox.y - origin.y)*divy;
		tymax = (minBox.y - origin.y)*divy;
	}

	if ( (tmin > tymax) || (tymin > tmax) )
		return false;

	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	float divz = 1.0f / dir.z;
	if (divz >= 0.0f)
	{
		tzmin = (minBox.z - origin.z)*divz;
		tzmax = (maxBox.z - origin.z)*divz;
	}
	else
	{
		tzmin = (maxBox.z - origin.z)*divz;
		tzmax = (minBox.z - origin.z)*divz;
	}

	if ( (tmin > tzmax) || (tzmin > tmax) )
		return false;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if (fabsf(tmax -tmin) < EPS)
		return false;

	if (tmin<0.0f)
	 	*tnear=0.0f;
	else
		*tnear=tmin;

	*tfar=tmax;

	return *tnear < *tfar;

}

__device__ bool _cuda_searchNextChildrenValidAndHit(index_node_t * elements, int size, int3 realDim, float3 origin, float3 ray, index_node_t father, float cTnear, float cTfar, int nLevels, int level, int3 minB, index_node_t * child, float * childTnear, float * childTfar)
{
	index_node_t childrenID = father << 3;
	int dim = (1<<(nLevels-level));
	int3 minBox = make_int3(minB.x, minB.y, minB.z);

	float closer = 0x7ff0000000000000;	//infinity
	bool find = false;
	float childTnearT = 0xfff0000000000000; // -infinity
	float childTfarT  = 0xfff0000000000000; // -infinity

	if (size==2)
	{
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.y+=dim;
		minBox.z-=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.x+=dim;
		minBox.y-=dim;
		minBox.z-=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.y+=dim;
		minBox.z-=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) && childTnearT >= cTnear && childTnearT <= closer &&
			_cuda_checkRangeGrid(elements, childrenID,0,1))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
	}
	else
	{
		unsigned int closer1 = _cuda_binary_search_closer_Grid(elements, childrenID,   0, size-1);
		unsigned int closer8 = _cuda_binary_search_closer_Grid(elements, childrenID+7, closer1, size-1) + 1;

		if (closer8 >= size)
			closer8 = size-1;

		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.y+=dim;
		minBox.z-=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.x+=dim;
		minBox.y-=dim;
		minBox.z-=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.y+=dim;
		minBox.z-=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
		minBox.z+=dim;
		if (_cuda_RayAABB2(origin, ray,  &childTnearT, &childTfarT, nLevels, minBox, level, realDim) &&  childTnearT>=cTnear && childTnearT <= closer &&
			_cuda_searchSecuentialGrid(elements, childrenID, closer1, closer8))
		{
			*child = childrenID;
			*childTnear = childTnearT;
			*childTfar = childTfarT;
			closer = childTnearT;
			find = true;
		}
		childrenID++;
	}

	return find;
}

__device__ int3 _cuda_updateCoordinatesGrid(int maxLevel, int cLevel, index_node_t cIndex, int nLevel, index_node_t nIndex, int3 minBox)
{
	if ( 0 == nIndex)
	{
		return make_int3(0,0,0);
	}
	else if (cLevel < nLevel)
	{
		index_node_t mask = (index_node_t) 1;
		minBox.z +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.y +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.x +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		return minBox;

	}
	else if (cLevel > nLevel)
	{
		return	getMinBoxIndex2(nIndex, nLevel, maxLevel);
	}
	else
	{
		index_node_t mask = (index_node_t)1;
		minBox.z +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.y +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.x +=  (nIndex & mask) << (maxLevel-nLevel); nIndex>>=1;
		minBox.z -=  (cIndex & mask) << (maxLevel-cLevel); cIndex>>=1;
		minBox.y -=  (cIndex & mask) << (maxLevel-cLevel); cIndex>>=1;
		minBox.x -=  (cIndex & mask) << (maxLevel-cLevel); cIndex>>=1;
		return minBox;
	}
}

__global__ void cuda_getFirtsVoxel(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float3 LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int finalLevel, visibleCube_t * p_indexNode, int numElements, int3 realDim)
{
	int i = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;

	if (i < numElements)
	{
    	int is = i % pvpW;
		int js = i / pvpW;

		float3 ray = LB - origin;
		ray += (js*h)*up + (is*w)*right;
		ray = normalize(ray);

		visibleCube_t * indexNode	= &p_indexNode[i];

		if (indexNode->state ==  CUDA_PAINTED)
		{
			indexNode->state = CUDA_DONE;
			return;
		}
		else if (indexNode->state ==  CUDA_NOCUBE)
		{
			float			currentTnear	= 0.0f;
			float			currentTfar		= 0.0f;
			index_node_t 	current			= indexNode->id == 0 ? 1 : indexNode->id;
			int				currentLevel	= 0;

			// Update tnear and tfar
			if (!_cuda_RayAABB(current, origin, ray,  &currentTnear, &currentTfar, nLevels, realDim) || currentTfar < 0.0f)
			{
				// NO CUBE FOUND
				indexNode->id 	= 0;
				return;
			}
			if (current != 1)
			{
				current  >>= 3;
				currentLevel = finalLevel - 1;
				currentTnear = currentTfar;
				//printf("--> %d %lld %d %f %f\n", i, current, currentLevel, currentTnear, currentTfar);
			}

			int3		minBox 		= getMinBoxIndex2(current, currentLevel, nLevels);

			while(1)
			{
				if (currentLevel == finalLevel)
				{
					indexNode->id = current;
					indexNode->state = CUDA_CUBE;
					return;
				}

				// Get fitst child >= currentTnear away
				index_node_t	child;
				float			childTnear;
				float			childTfar;
				if (_cuda_searchNextChildrenValidAndHit(octree[currentLevel+1], sizes[currentLevel+1], realDim, origin, ray, current, currentTnear, currentTfar, nLevels, currentLevel+1, minBox, &child, &childTnear, &childTfar))
				{
					minBox = _cuda_updateCoordinatesGrid(nLevels, currentLevel, current, currentLevel + 1, child, minBox);
					current = child;
					currentLevel++;
					currentTnear = childTnear;
					currentTfar = childTfar;
				//if (currentTnear == currentTfar)
				//	printf("--> %d %lld %d %f %f\n", i, current, currentLevel, currentTnear, currentTfar);
				}
				else if (current == 1) 
				{
					indexNode->id 	= 0;
					return;
				}
				else
				{
					minBox = _cuda_updateCoordinatesGrid(nLevels, currentLevel, current, currentLevel - 1, current >> 3, minBox);
					current >>= 3;
					currentLevel--;
					currentTnear = currentTfar;
				}

			}
		}
	}
	return;
}

__global__ void cuda_drawCubes(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float3 LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int finalLevel, int numElements, int3 realDim, float maxHeight, float * r, float * g, float * b, float * screen)
{
	int i = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;

	if (i < numElements)
	{
    	int is = i % pvpW;
		int js = i / pvpW;

		float3 ray = LB - origin;
		ray += (js*h)*up + (is*w)*right;
		ray = normalize(ray);

		float			currentTnear	= 0.0f;
		float			currentTfar		= 0.0f;
		index_node_t 	current			= 1;
		int				currentLevel	= 0;

		// Update tnear and tfar
		if (!_cuda_RayAABB(current, origin, ray,  &currentTnear, &currentTfar, nLevels, realDim) || currentTfar < 0.0f)
		{
			// NO CUBE FOUND
			screen[i*3] = r[NUM_COLORS];
			screen[i*3+1] = g[NUM_COLORS];
			screen[i*3+2] = b[NUM_COLORS];
			return;
		}
		if (current != 1)
		{
			current  >>= 3;
			currentLevel = finalLevel - 1;
			currentTnear = currentTfar;
		}

		int3		minBox 		= getMinBoxIndex2(current, currentLevel, nLevels);

		while(currentLevel != finalLevel)
		{
			// Get fitst child >= currentTnear away
			index_node_t	child;
			float			childTnear;
			float			childTfar;
			if (_cuda_searchNextChildrenValidAndHit(octree[currentLevel+1], sizes[currentLevel+1], realDim, origin, ray, current, currentTnear, currentTfar, nLevels, currentLevel+1, minBox, &child, &childTnear, &childTfar))
			{
				minBox = _cuda_updateCoordinatesGrid(nLevels, currentLevel, current, currentLevel + 1, child, minBox);
				current = child;
				currentLevel++;
				currentTnear = childTnear;
				currentTfar = childTfar;
			}
			else if (current == 1) 
			{
				screen[i*3] = r[NUM_COLORS];
				screen[i*3+1] = g[NUM_COLORS];
				screen[i*3+2] = b[NUM_COLORS];
				return;
			}
			else
			{
				minBox = _cuda_updateCoordinatesGrid(nLevels, currentLevel, current, currentLevel - 1, current >> 3, minBox);
				current >>= 3;
				currentLevel--;
				currentTnear = currentTfar;
			}

		}
		int dim = 1 << (3*(nLevels - currentLevel));
		float3 minBoxC = _cuda_BoxToCoordinates(minBox , realDim);
		float3 maxBoxC = _cuda_BoxToCoordinates(minBox + make_int3(dim,dim,dim), realDim);
		float3 n = make_float3(0.0f,0.0f,0.0f);
		float3 hit = origin + ray*currentTnear;
		float aux = 0.0f;

		if (fabsf(maxBoxC.x - origin.x) < fabsf(minBoxC.x - origin.x))
		{
			aux = minBoxC.x;
			minBoxC.x = maxBoxC.x; 
			maxBoxC.x = aux;
		}
		if (fabsf(maxBoxC.y - origin.y) < fabsf(minBoxC.y - origin.y))
		{
			aux = minBoxC.y;
			minBoxC.y = maxBoxC.y; 
			maxBoxC.y = aux;
		}
		if (fabsf(maxBoxC.z - origin.z) < fabsf(minBoxC.z - origin.z))
		{
			aux = minBoxC.z;
			minBoxC.z = maxBoxC.z; 
			maxBoxC.z = aux;
		}

		if(fabsf(hit.x - minBoxC.x) < EPS) 
			n.x = -1.0f;
		else if(fabsf(hit.x - maxBoxC.x) < EPS) 
			n.x = 1.0f;
		else if(fabsf(hit.y - minBoxC.y) < EPS) 
			n.y = -1.0f;
		else if(fabsf(hit.y - maxBoxC.y) < EPS) 
			n.y = 1.0f;
		else if(fabsf(hit.z - minBoxC.z) < EPS) 
			n.z = -1.0f;
		else if(fabsf(hit.z - maxBoxC.z) < EPS) 
			n.z = 1.0f;


		float3 l = hit - origin;// ligth; light on the camera
		l = normalize(l);	
		float dif = fabsf(n.x*l.x + n.y*l.y + n.z*l.z);

		float a = hit.y/maxHeight;
		int pa = floorf(a*NUM_COLORS);
		if (pa < 0)
		{
			screen[i*3]   =r[0]*dif;
			screen[i*3+1] =g[0]*dif;
			screen[i*3+2] =b[0]*dif;
		}
		else if (pa >= NUM_COLORS-1) 
		{
			screen[i*3]   = r[NUM_COLORS-1]*dif;
			screen[i*3+1] = g[NUM_COLORS-1]*dif;
			screen[i*3+2] = b[NUM_COLORS-1]*dif;
		}
		else
		{
			float dx = (a*(float)NUM_COLORS - (float)pa);
			screen[i*3]   = (r[pa] + (r[pa+1]-r[pa])*dx)*dif;
			screen[i*3+1] = (g[pa] + (g[pa+1]-g[pa])*dx)*dif;
			screen[i*3+2] = (b[pa] + (b[pa+1]-b[pa])*dx)*dif;
		}
	}
}

/*
 ******************************************************************************************************
 ************ METHODS OCTREEMCUDA *********************************************************************
 ******************************************************************************************************
 */

	void getBoxIntersectedOctree(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float3 LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int finalLevel, int numElements, visibleCubeGPU_t visibleGPU, int3 realDim, cudaStream_t stream)
{

	dim3 threads = getThreads(numElements);
	dim3 blocks = getBlocks(numElements);

	cuda_getFirtsVoxel<<<blocks,threads, 0, stream>>>(octree, sizes, nLevels, origin, LB, up, right, w, h, pvpW, pvpH, finalLevel, visibleGPU, numElements, realDim);

}

	void drawCubes(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float3 LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int finalLevel, int numElements, int3 realDim, float maxHeight, float * r, float * g, float * b, float * screen, cudaStream_t stream)
{
	dim3 threads = getThreads(numElements);
	dim3 blocks = getBlocks(numElements);

	cuda_drawCubes<<<blocks,threads, 0, stream>>>(octree, sizes, nLevels, origin, LB, up, right, w, h, pvpW, pvpH, finalLevel, numElements, realDim, maxHeight, r, g, b, screen);

	if (cudaSuccess != cudaStreamSynchronize(stream))
	{
		std::cerr<<"Error octree: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
}

/*
 ******************************************************************************************************
 ************ METHODS OCTREE CUDA CREATE **************************************************************
 ******************************************************************************************************
 */

__global__ void cuda_insertOctreePointers(index_node_t ** octreeGPU, int * sizes, index_node_t * memoryGPU)
{
	int offset = 0;
	for(int i=0;i<threadIdx.x; i++)
		offset+=sizes[i];

	octreeGPU[threadIdx.x] = &memoryGPU[offset];
}


void insertOctreePointers(index_node_t ** octreeGPU, int * sizes, index_node_t * memoryGPU, int levels)
{
	dim3 blocks(1);
	dim3 threads(levels);

	cuda_insertOctreePointers<<<blocks,threads>>>(octreeGPU, sizes, memoryGPU);
	if (cudaSuccess != cudaDeviceSynchronize())
	{
		std::cerr<<"Error init octree: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
}

}
