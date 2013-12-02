#ifndef DEVICE_CODE
#include <../src/textures.cu>
#endif

#include "mortonCodeUtil.h"
#include "cuda_help.h"

#include <cutil_math.h>

#include <iostream>
#include <fstream>

#define posToIndex(i,j,k,d) ((k)+(j)*(d)+(i)*(d)*(d))

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

inline __device__ int _cuda_searchCoordinateX(float x, int min, int max)
{
	for(int i=min + CUBE_INC; i<max + CUBE_INC; i++)
		if (tex1Dfetch(xgrid,i) <= x && x < tex1Dfetch(xgrid, i+1))
			return i - CUBE_INC;
	
	return -10;
}

inline __device__ int _cuda_searchCoordinateY(float x, int min, int max)
{
	for(int i=min + CUBE_INC; i<max + CUBE_INC; i++)
		if (tex1Dfetch(ygrid,i) <= x && x < tex1Dfetch(ygrid, i+1))
			return i - CUBE_INC;
	
	return -10;
}

inline __device__ int _cuda_searchCoordinateZ(float x, int min, int max)
{
	for(int i=min + CUBE_INC; i<max + CUBE_INC; i++)
		if (tex1Dfetch(zgrid,i) <= x && x < tex1Dfetch(zgrid, i+1))
			return i - CUBE_INC;
	
	return -10;
}

inline __device__ bool _cuda_RayAABB(float3 origin, float3 dir,  float * tnear, float * tfar, float3 minBox, float3 maxBox)
{
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

inline __device__ float getElement(int x, int y, int z, float * data, int3 dim)
{
		return data[posToIndex(x,y,z,dim.x)];
}

__device__ float getElementInterpolateGrid(float3 pos, float * data, int3 minBox, int3 dim)
{
	float3 posR;
	float3 pi = make_float3(modff(pos.x,&posR.x), modff(pos.y,&posR.y), modff(pos.z,&posR.z));

	int x0 = posR.x - minBox.x;
	int y0 = posR.y - minBox.y;
	int z0 = posR.z - minBox.z;
	int x1 = x0 + 1;
	int y1 = y0 + 1;
	int z1 = z0 + 1;

	float c00 = getElement(x0,y0,z0,data,dim) * (1.0f - pi.x) + getElement(x1,y0,z0,data,dim) * pi.x;
	float c01 = getElement(x0,y0,z1,data,dim) * (1.0f - pi.x) + getElement(x1,y0,z1,data,dim) * pi.x;
	float c10 = getElement(x0,y1,z0,data,dim) * (1.0f - pi.x) + getElement(x1,y1,z0,data,dim) * pi.x;
	float c11 = getElement(x0,y1,z1,data,dim) * (1.0f - pi.x) + getElement(x1,y1,z1,data,dim) * pi.x;

	float c0 = c00 * (1.0f - pi.y) + c10 * pi.y;
	float c1 = c01 * (1.0f - pi.y) + c11 * pi.y;

	return c0 * (1.0f - pi.z) + c1 * pi.z;
}

inline __device__ float3 getNormal(float3 pos, float * data, int3 minBox, int3 maxBox)
{
	return normalize(make_float3(	
				(getElementInterpolateGrid(make_float3(pos.x-1.0f,pos.y,pos.z),data,minBox,maxBox) - getElementInterpolateGrid(make_float3(pos.x+1.0f,pos.y,pos.z),data,minBox,maxBox))        /2.0f,
				(getElementInterpolateGrid(make_float3(pos.x,pos.y-1,pos.z),data,minBox,maxBox) - getElementInterpolateGrid(make_float3(pos.x,pos.y+1.0f,pos.z),data,minBox,maxBox))        /2.0f,
				(getElementInterpolateGrid(make_float3(pos.x,pos.y,pos.z-1),data,minBox,maxBox) - getElementInterpolateGrid(make_float3(pos.x,pos.y,pos.z+1.0f),data,minBox,maxBox))        /2.0f));
}

}
