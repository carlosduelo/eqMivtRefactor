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
	r.x = pos.x >= realDim.x ? tex1Dfetch(xgrid, CUBE_INC + realDim.x-1) + pos-x - realDim.x : tex1Dfetch(xgrid, CUBE_INC + pos.x);
	r.y = pos.y >= realDim.y ? tex1Dfetch(ygrid, CUBE_INC + realDim.y-1) + pos-x - realDim.x : tex1Dfetch(ygrid, CUBE_INC + pos.y);
	r.z = pos.z >= realDim.z ? tex1Dfetch(zgrid, CUBE_INC + realDim.z-1) + pos-x - realDim.x : tex1Dfetch(zgrid, CUBE_INC + pos.z);

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
		return elements[max] >= index && elements[min] <= index;
}

__device__ int _cuda_binary_search_closer_Grid(index_node_t * elements, index_node_t index, int min, int max)
{
	int middle = 0;
	while(1)
	{
		int diff 	= max-min;
		middle	= min + (diff / 2);
		if (middle % 2 == 1) middle--;

		if (diff <= 1) return middle;
		if (elements[middle+1] >= index && elements[middle] <= index) return middle;
		if (index < elements[middle])
			max = middle-1;
		else //(index > elements[middle+1])
			min = middle + 2;
	}
}

__device__  bool _cuda_searchSecuentialGrid(index_node_t * elements, index_node_t index, int min, int max)
{
	for(int i=min; i<max; i+=2)
	{
		if (elements[i] > index)
			return false;
		if (elements[i+1] >= index && elements[i] <= index)
			return true;
	}

	return false;
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
#if 0
	int dimC = 1 << (nLevels - level);
	int dimF = dimC << 1;
	float3 minBoxC = _cuda_BoxToCoordinates(minB, realDim);	
	float3 midBoxC = _cuda_BoxToCoordinates(minB + make_int3(dimC, dimC, dimC), realDim);	
	float3 maxBoxC = _cuda_BoxToCoordinates(minB + make_int3(dimF, dimF, dimF), realDim);	
	float3 tmin, tmid, tmax;
	float3 div = make_float3(1.0f / ray.x, 1.0f / ray.y, 1.0f / ray.z); 
	index_node_t mask = 0;

	if (div.x >= 0)
	{
		tmin.x = minBoxC.x - origin.x;
		tmid.x = midBoxC.x - origin.x;
		tmax.x = maxBoxC.x - origin.x;
	}
	else
	{
		tmin.x = maxBoxC.x - origin.x;
		tmid.x = midBoxC.x - origin.x;
		tmax.x = minBoxC.x - origin.x;
		mask |= 4;
	}
	if (div.y >= 0)
	{
		tmin.y = minBoxC.y - origin.y;
		tmid.y = midBoxC.y - origin.y;
		tmax.y = maxBoxC.y - origin.y;
	}                     
	else                  
	{                     
		tmin.y = maxBoxC.y - origin.y;
		tmid.y = midBoxC.y - origin.y;
		tmax.y = minBoxC.y - origin.y;
		mask |= 2;
	}
	if (div.z >= 0)
	{
		tmin.z = minBoxC.z - origin.x;
		tmid.z = midBoxC.z - origin.x;
		tmax.z = maxBoxC.z - origin.x;
	}                     
	else                  
	{                     
		tmin.z = maxBoxC.z - origin.z;
		tmid.z = midBoxC.z - origin.z;
		tmax.z = minBoxC.z - origin.z;
		mask |= 1;
	}

	tmin = tmin * div;
	tmid = tmid * div;
	tmax = tmax * div;

	index_node_t c = 0;
	index_node_t childrenID = father << 3;
	unsigned int closer1 = 0;
	unsigned int closer8 = size;

	if (size != 2)
	{
		closer1 =  _cuda_binary_search_closer_Grid(elements , childrenID , 0, size-1);
	}

	float text = 0.0f;
	float tent = 0.0f;
	float closer = 0x7ff0000000000000;	//infinity
	bool find = false;

	c |= tent > tmid.x && tent < tmax.x ? 4 : 0;
	c |= tent > tmid.y && tent < tmax.y ? 2 : 0;
	c |= tent > tmid.z && tent < tmax.z ? 1 : 0;

	switch(c)
	{
		case 0:
				tent = fmaxf(tmin.x, fmaxf(tmin.y, tmin.z));
				text = fminf(tmid.x, fminf(tmid.y, tmid.z));
			break;  
		case 1:
				tent = fmaxf(tmin.x, fmaxf(tmin.y, tmid.z));
				text = fminf(tmid.x, fminf(tmid.y, tmax.z));
			break;  
		case 2:
				tent = fmaxf(tmin.x, fmaxf(tmid.y, tmin.z));
				text = fminf(tmid.x, fminf(tmax.y, tmid.z));
			break;  
		case 3:
				tent = fmaxf(tmin.x, fmaxf(tmid.y, tmid.z));
				text = fminf(tmid.x, fminf(tmax.y, tmax.z));
			break;  
		case 4:
				tent = fmaxf(tmid.x, fmaxf(tmin.y, tmin.z));
				text = fminf(tmax.x, fminf(tmid.y, tmid.z));
			break;  
		case 5:
				tent = fmaxf(tmid.x, fmaxf(tmin.y, tmid.z));
				text = fminf(tmax.x, fminf(tmid.y, tmax.z));
			break;  
		case 6:
				tent = fmaxf(tmid.x, fmaxf(tmid.y, tmin.z));
				text = fminf(tmax.x, fminf(tmax.y, tmid.z));
			break;  
		case 7:
				tent = fmaxf(tmid.x, fmaxf(tmid.y, tmid.z));
				text = fminf(tmax.x, fminf(tmax.y, tmax.z));
			break;  
	}

	do
	{

		if (tent < text && 
			tent >= cTnear && 
			tent <= closer && 
			_cuda_searchSecuentialGrid(elements, childrenID | (c^mask), closer1, closer8))
		{
			*child = childrenID | (c ^ mask);
			*childTnear = tent;
			*childTfar = text;
			closer = tent;
			find = true;
		}

		float te = 0.0f;
		switch(c)
		{
			case 0:
				te = fminf(tmid.x, fminf(tmid.y, tmid.z));
				if (te == tmid.x)
				{
					c=4;
					tent = fmaxf(tmid.x, fmaxf(tmin.y, tmin.z));
					text = fminf(tmax.x, fminf(tmid.y, tmid.z));
				}
				else if (te == tmid.y)
				{
					c=2;
					tent = fmaxf(tmin.x, fmaxf(tmid.y, tmin.z));
					text = fminf(tmid.x, fminf(tmax.y, tmid.z));
				}
				else if (te == tmid.z)
				{
					c=1;
					tent = fmaxf(tmin.x, fmaxf(tmin.y, tmid.z));
					text = fminf(tmid.x, fminf(tmid.y, tmax.z));
				}
				break;  
			case 1:	
				te = fminf(tmid.x, fminf(tmid.y, tmax.z));
				if (te == tmid.x)
				{
					c=5;
					tent = fmaxf(tmid.x, fmaxf(tmin.y, tmid.z));
					text = fminf(tmax.x, fminf(tmid.y, tmax.z));
				}
				else if (te == tmid.y)
				{
					c=3;
					tent = fmaxf(tmin.x, fmaxf(tmid.y, tmid.z));
					text = fminf(tmid.x, fminf(tmax.y, tmax.z));
				}
				else
					c=8;
				break;  
			case 2:	
				te = fminf(tmid.x, fminf(tmax.y, tmid.z));
				if (te == tmid.x) 
				{
					c=6;
					tent = fmaxf(tmid.x, fmaxf(tmid.y, tmin.z));
					text = fminf(tmax.x, fminf(tmax.y, tmid.z));
				}
				else if (te == tmid.z)
				{
					c=3;
					tent = fmaxf(tmin.x, fmaxf(tmid.y, tmid.z));
					text = fminf(tmid.x, fminf(tmax.y, tmax.z));
				}
				else
					c=8;
				break;  
			case 3:	
				te = fminf(tmid.x, fminf(tmax.y, tmax.z));
				if (te == tmid.x) 
				{
					c=7;
					tent = fmaxf(tmid.x, fmaxf(tmid.y, tmid.z));
					text = fminf(tmax.x, fminf(tmax.y, tmax.z));
				}
				else 
					c=8;
				break;  
			case 4:	
				te = fminf(tmax.x, fminf(tmid.y, tmid.z));
				if (te == tmid.y)
				{
					c=6;
					tent = fmaxf(tmid.x, fmaxf(tmid.y, tmin.z));
					text = fminf(tmax.x, fminf(tmax.y, tmid.z));
				}
				else if (te == tmid.z)
				{
					c=5;
					tent = fmaxf(tmid.x, fmaxf(tmin.y, tmid.z));
					text = fminf(tmax.x, fminf(tmid.y, tmax.z));
				}
				else
					c=8;
				break;  
			case 5:	
				te = fminf(tmax.x, fminf(tmid.y, tmax.z));
				if (te == tmid.y)
				{
					c=7;
					tent = fmaxf(tmid.x, fmaxf(tmid.y, tmid.z));
					text = fminf(tmax.x, fminf(tmax.y, tmax.z));
				}
				else 
					c=8;
				break;  
			case 6:	
				te = fminf(tmax.x, fminf(tmax.y, tmid.z));
				if (te == tmid.z) 
				{
					c=7;
					tent = fmaxf(tmid.x, fmaxf(tmid.y, tmid.z));
					text = fminf(tmax.x, fminf(tmax.y, tmax.z));
				}
				else
					c=8;
				break;  
			case 7:
				c = 8;
				break;  
		}
	}
	while(c < 8);

	return find;

#else
	index_node_t childrenID = father << 3;
	int dim = (1<<(nLevels-level));
	float closer = 0x7ff0000000000000;	//infinity
	bool find = false;
	float childTnearT = 0xfff0000000000000; // -infinity
	float childTfarT  = 0xfff0000000000000; // -infinity
	int3 minBox = minB;

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
		unsigned int closer8 = size;

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
	#endif
}

__device__ int3 _cuda_updateCoordinatesGrid(int maxLevel, int cLevel, index_node_t cIndex, int nLevel, index_node_t nIndex, int3 minBox)
{
	if ( 0 == nIndex)
	{
		return make_int3(0,0,0);
	}
	else if (cLevel < nLevel)
	{
		int dim = 1 << (maxLevel-nLevel);
		minBox.z +=  (nIndex & 0x1) * dim; nIndex>>=1;
		minBox.y +=  (nIndex & 0x1) * dim; nIndex>>=1;
		minBox.x +=  (nIndex & 0x1) * dim;
		return minBox;

	}
	else if (cLevel > nLevel)
	{
		return	getMinBoxIndex2(nIndex, nLevel, maxLevel);
	}
	else
	{
		int dim = 1 << (maxLevel-nLevel);
		minBox.z +=  (nIndex & 0x1) * dim; nIndex>>=1;
		minBox.y +=  (nIndex & 0x1) * dim; nIndex>>=1;
		minBox.x +=  (nIndex & 0x1) * dim;
		minBox.z -=  (cIndex & 0x1) * dim; cIndex>>=1;
		minBox.y -=  (cIndex & 0x1) * dim; cIndex>>=1;
		minBox.x -=  (cIndex & 0x1) * dim;
		return minBox;
	}
}

__device__ bool _cuda_octreeIteration(index_node_t ** octree, int * sizes, float3 origin, float3 ray, int nLevels, int finalLevel, visibleCube_t * indexNode, int3 realDim, float * currentTnear, float * currentTfar)
{
	*currentTnear	= 0.0f;
	*currentTfar	= 0.0f;
	index_node_t 	current			= indexNode->id == 0 ? 1 : indexNode->id;
	int				currentLevel	= 0;

	// Update tnear and tfar
	if (!_cuda_RayAABB(current, origin, ray,  currentTnear, currentTfar, nLevels, realDim) || (*currentTfar) < 0.0f)
	{
		// NO CUBE FOUND
		indexNode->state = CUDA_NOCUBE;
		return false;
	}
	if (current != 1)
	{
		current  >>= 3;
		currentLevel = finalLevel - 1;
		*currentTnear = *currentTfar;
	}

	int3		minBox 		= getMinBoxIndex2(current, currentLevel, nLevels);

	while(1)
	{
		if (currentLevel == finalLevel)
		{
			indexNode->id = current;
			indexNode->state = CUDA_CUBE;
			return true;
		}

		// Get fitst child >= currentTnear away
		index_node_t	child;
		float			childTnear;
		float			childTfar;
		if (_cuda_searchNextChildrenValidAndHit(octree[currentLevel+1], sizes[currentLevel+1], realDim, origin, ray, current, *currentTnear, *currentTfar, nLevels, currentLevel+1, minBox, &child, &childTnear, &childTfar))
		{
			minBox = _cuda_updateCoordinatesGrid(nLevels, currentLevel, current, currentLevel + 1, child, minBox);
			current = child;
			currentLevel++;
			*currentTnear = childTnear;
			*currentTfar = childTfar;
		}
		else if (current == 1) 
		{
			indexNode->state = CUDA_NOCUBE;
			return false;
		}
		else
		{
			minBox = _cuda_updateCoordinatesGrid(nLevels, currentLevel, current, currentLevel - 1, current >> 3, minBox);
			current >>= 3;
			currentLevel--;
			*currentTnear = *currentTfar;
		}

	}
}

__device__ bool _cuda_rayCaster(float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int numRays, float iso, visibleCube_t * cube, int levelO, int levelC, int nLevel, float maxHeight, int3 realDim, float * r, float * g, float * b, float * screen, int offset, float * data)
{
	unsigned int tid = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < numRays)
	{
		tid += offset;
		float tnear;
		float tfar;
		// To do test intersection real cube position
		int3 minBox = getMinBoxIndex2(cube->id, levelO, nLevel);
		int dim = 1 << (nLevel-levelO);
		//int dim = powf(2,nLevel-levelO);
		int3 maxBox = minBox + make_int3(dim,dim,dim);
		float3 minBoxC = _cuda_BoxToCoordinates(minBox, realDim);
		float3 maxBoxC = _cuda_BoxToCoordinates(maxBox, realDim);

		int i = tid % pvpW;
		int j = tid / pvpW;

		float3 ray = LB - origin;
		ray += (j*h)*up + (i*w)*right;
		ray = normalize(ray);

		if  (_cuda_RayAABB(origin, ray,  &tnear, &tfar, minBoxC, maxBoxC))
		{
			// To ray caster is needed bigger cube, so add cube inc
			int3 minBoxD = getMinBoxIndex2(cube->id >> (3*(levelO - levelC)), levelC, nLevel) - make_int3(CUBE_INC, CUBE_INC, CUBE_INC);
			int3 dimD;
			dimD.x = (1 << (nLevel-levelC)) + 2*CUBE_INC;
			dimD.y = (1 << (nLevel-levelC)) + 2*CUBE_INC;
			dimD.z = (1 << (nLevel-levelC)) + 2*CUBE_INC;

			float3 Xnear = origin + tnear * ray;

			int3 pos = make_int3(	_cuda_searchCoordinateX(Xnear.x, minBox.x - 1, maxBox.x+1),
									_cuda_searchCoordinateY(Xnear.y, minBox.y - 1, maxBox.y+1),
									_cuda_searchCoordinateZ(Xnear.z, minBox.z - 1, maxBox.z+1));

			bool hit = false;
			float3 Xfar = Xnear;
			float3 Xnew = Xnear;
			bool primera 	= true;
			float ant		= 0.0f;
			float sig		= 0.0f;

			while (!hit &&
				(minBox.x-1 <= pos.x && pos.x <= maxBox.x) &&
				(minBox.y-1 <= pos.y && pos.y <= maxBox.y) &&
				(minBox.z-1 <= pos.z && pos.z <= maxBox.z))
			{
				if (pos.x >= 0 && pos.y >= 0 && pos.z >= 0 && pos.x < realDim.x-1 && pos.y < realDim.y-1 && pos.z < realDim.z-1)
				{
					float3 xyz = make_float3(	pos.x + ((Xnear.x - tex1Dfetch(xgrid, pos.x + CUBE_INC)) / (tex1Dfetch(xgrid, pos.x+1 + CUBE_INC) - tex1Dfetch(xgrid, pos.x + CUBE_INC))),
												pos.y + ((Xnear.y - tex1Dfetch(ygrid, pos.y + CUBE_INC)) / (tex1Dfetch(ygrid, pos.y+1 + CUBE_INC) - tex1Dfetch(ygrid, pos.y + CUBE_INC))),
												pos.z + ((Xnear.z - tex1Dfetch(zgrid, pos.z + CUBE_INC)) / (tex1Dfetch(zgrid, pos.z+1 + CUBE_INC) - tex1Dfetch(zgrid, pos.z + CUBE_INC))));
					
					if (primera)
					{
						ant = getElementInterpolateGrid(xyz, data, minBoxD, dimD);
						Xfar = Xnear;
						primera = false;
					}
					else
					{
						sig = getElementInterpolateGrid(xyz, data, minBoxD, dimD);

						if (( ((iso-ant)<0.0f) && ((iso-sig)<0.0f)) || ( ((iso-ant)>0.0f) && ((iso-sig)>0.0)))
						{
							ant = sig;
							Xfar=Xnear;
						}
						else
						{
							float a = (iso-ant)/(sig-ant);
							Xnew = Xfar*(1.0f-a)+ Xnear*a;
							hit = true;
						}
					}
				}

				// Update Xnear
				Xnear += ((fminf(fabs(tex1Dfetch(xgrid, pos.x+1 + CUBE_INC) - tex1Dfetch(xgrid, pos.x + CUBE_INC)), fminf(fabs(tex1Dfetch(ygrid, pos.y+1 + CUBE_INC) - tex1Dfetch(ygrid, pos.y + CUBE_INC)),fabs( tex1Dfetch(zgrid, pos.z+1 + CUBE_INC) - tex1Dfetch(zgrid, pos.z + CUBE_INC))))) / 3.0f) * ray;

				// Get new pos
				while((minBox.x-2 <= pos.x && pos.x <= maxBox.x + 1) &&  !(tex1Dfetch(xgrid, pos.x + CUBE_INC) <= Xnear.x && Xnear.x < tex1Dfetch(xgrid, pos.x+1 + CUBE_INC)))
					pos.x = ray.x < 0 ? pos.x - 1 : pos.x +1;
				while((minBox.y-2 <= pos.y && pos.y <= maxBox.y + 1) &&!(tex1Dfetch(ygrid, pos.y + CUBE_INC) <= Xnear.y && Xnear.y < tex1Dfetch(ygrid, pos.y+1 + CUBE_INC)))
					pos.y = ray.y < 0 ? pos.y - 1 : pos.y +1;
				while((minBox.z-2 <= pos.z && pos.z <= maxBox.z + 1) &&!(tex1Dfetch(zgrid, pos.z + CUBE_INC) <= Xnear.z && Xnear.z < tex1Dfetch(zgrid, pos.z+1 + CUBE_INC)))
					pos.z = ray.z < 0 ? pos.z - 1 : pos.z +1;
			}

			if (hit)
			{
				pos = make_int3(	_cuda_searchCoordinateX(Xnew.x, minBox.x - 1, maxBox.x+1),
									_cuda_searchCoordinateY(Xnew.y, minBox.y - 1, maxBox.y+1),
									_cuda_searchCoordinateZ(Xnew.z, minBox.z - 1, maxBox.z+1));

				float3 xyz = make_float3(	pos.x + ((Xnew.x - tex1Dfetch(xgrid, pos.x + CUBE_INC)) / (tex1Dfetch(xgrid, pos.x+1 + CUBE_INC) - tex1Dfetch(xgrid, pos.x + CUBE_INC))),
											pos.y + ((Xnew.y - tex1Dfetch(ygrid, pos.y + CUBE_INC)) / (tex1Dfetch(ygrid, pos.y+1 + CUBE_INC) - tex1Dfetch(ygrid, pos.y + CUBE_INC))),
											pos.z + ((Xnew.z - tex1Dfetch(zgrid, pos.z + CUBE_INC)) / (tex1Dfetch(zgrid, pos.z+1 + CUBE_INC) - tex1Dfetch(zgrid, pos.z + CUBE_INC))));

				float3 n = getNormal(xyz, data, minBoxD,  dimD);
				float3 l = Xnew - origin;// ligth; light on the camera
				l = normalize(l);	
				float dif = fabsf(n.x*l.x + n.y*l.y + n.z*l.z);
				float a = Xnew.y/maxHeight;
				int pa = floorf(a*NUM_COLORS);
				if (pa < 0)
				{
					screen[tid*3]   =r[0]*dif;
					screen[tid*3+1] =g[0]*dif;
					screen[tid*3+2] =b[0]*dif;
				}
				else if (pa >= NUM_COLORS-1) 
				{
					screen[tid*3]   = r[NUM_COLORS-1]*dif;
					screen[tid*3+1] = g[NUM_COLORS-1]*dif;
					screen[tid*3+2] = b[NUM_COLORS-1]*dif;
				}
				else
				{
					float dx = (a*(float)NUM_COLORS - (float)pa);
					screen[tid*3]   = (r[pa] + (r[pa+1]-r[pa])*dx)*dif;
					screen[tid*3+1] = (g[pa] + (g[pa+1]-g[pa])*dx)*dif;
					screen[tid*3+2] = (b[pa] + (b[pa+1]-b[pa])*dx)*dif;
				}
				cube->state= CUDA_PAINTED;
				return true;
			}
			else
			{
				cube->state = CUDA_NOCUBE;
				return false;
			}
		}
		else
		{
			screen[tid*3] = 1.0f;
			screen[tid*3+1] = 0.0f;
			screen[tid*3+2] = 0.0f;
			cube->state = CUDA_PAINTED;
			return false;
		}
	}
	return false;
}


__global__ void cuda_getFirtsVoxel(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float3 LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int finalLevel, int levelCube, visibleCube_t * p_indexNode, int numElements, int offset, int3 realDim, float * r, float * g, float * b, float * pixelBuffer, float iso, float maxHeight, float ** tableCubes)
{
	int i = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;

	if (i < numElements)
	{
		float			currentTnear	= 0.0f;
		float			currentTfar		= 0.0f;

		i += offset;
    	int is = i % pvpW;
		int js = i / pvpW;

		float3 ray = LB - origin;
		ray += (js*h)*up + (is*w)*right;
		ray = normalize(ray);

		visibleCube_t * indexNode	= &p_indexNode[i];
		index_node_t minV = coordinateToIndex(make_int3(0,0,0), levelCube, nLevels); 

		if (indexNode->state == CUDA_CACHED)
		{
			_cuda_rayCaster(origin, LB, up, right, w, h, pvpW, pvpH, numElements, iso, indexNode, finalLevel, levelCube, nLevels, maxHeight, realDim, r, g, b, pixelBuffer, offset, tableCubes[indexNode->idCube - minV]);
		}

		if (indexNode->state ==  CUDA_NOCUBE)
		{
			while(1)
			{
				if (_cuda_octreeIteration(octree, sizes, origin, ray, nLevels, finalLevel, indexNode, realDim, &currentTnear, &currentTfar))
				{
					index_node_t idCubeN = indexNode->id >> (3*(finalLevel - levelCube));
					float * d = tableCubes[idCubeN - minV];

					if ( d != 0)
					{
						if (_cuda_rayCaster(origin, LB, up, right, w, h, pvpW, pvpH, numElements, iso, indexNode, finalLevel, levelCube, nLevels, maxHeight, realDim, r, g, b, pixelBuffer, offset, d))
						{
							return;
						}
						else
						{
							indexNode->state = CUDA_NOCUBE;
						}
					}
					else
					{
						indexNode->state = CUDA_CUBE;
						return;
					}
				}
				else
				{
					// NO CUBE FOUND
					pixelBuffer[i*3] = r[NUM_COLORS];
					pixelBuffer[i*3+1] = g[NUM_COLORS];
					pixelBuffer[i*3+2] = b[NUM_COLORS];
					indexNode->state = CUDA_PAINTED;
					return;
				}
			}
		}
	}
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
		visibleCube_t indexNode = {0};

		// Update tnear and tfar
		if (_cuda_octreeIteration(octree, sizes, origin, ray, nLevels, finalLevel, &indexNode, realDim, &currentTnear, &currentTfar))
		{
			int3 minBox = getMinBoxIndex2(indexNode.id, finalLevel, nLevels);
			int dim = 1 << (3*(nLevels - finalLevel));
			float3 minBoxC = _cuda_BoxToCoordinates(minBox , realDim);
			float3 maxBoxC = _cuda_BoxToCoordinates(minBox + make_int3(dim,dim,dim), realDim);
			float3 n = make_float3(0.0f,0.0f,0.0f);
			float3 hit = origin + ray*currentTnear;

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
		else
		{
			// NO CUBE FOUND
			screen[i*3] = r[NUM_COLORS];
			screen[i*3+1] = g[NUM_COLORS];
			screen[i*3+2] = b[NUM_COLORS];
		}
	}
}

/*
 ******************************************************************************************************
 ************ METHODS OCTREEMCUDA *********************************************************************
 ******************************************************************************************************
 */

void getBoxIntersectedOctree(index_node_t ** octree, int * sizes, int nLevels, float3 origin, float3 LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int finalLevel, int levelCube, int numElements, visibleCubeGPU_t visibleGPU, int offset, int3 realDim, float * r, float * g, float * b, float * pixelBuffer, float iso, float maxHeight, float ** tableCubes, cudaStream_t stream)
{

	dim3 threads = getThreads(numElements);
	dim3 blocks = getBlocks(numElements);

	cuda_getFirtsVoxel<<<blocks,threads, 0, stream>>>(octree, sizes, nLevels, origin, LB, up, right, w, h, pvpW, pvpH, finalLevel, levelCube, visibleGPU, numElements, offset, realDim, r, g, b, pixelBuffer, iso, maxHeight, tableCubes);

	#ifndef NDEBUG
	if (cudaSuccess != cudaStreamSynchronize(stream))
	{
		std::cerr<<"Error octree: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	#endif

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

	cuda_insertOctreePointers<<<blocks,threads,0, 0>>>(octreeGPU, sizes, memoryGPU);

	if (cudaSuccess != cudaStreamSynchronize(0))
	{
		std::cerr<<"Error init octree: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
}

}
