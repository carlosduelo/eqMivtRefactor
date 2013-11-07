#include "rayCaster_cuda.h"

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

__global__ void cuda_rayCaster(float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int numRays, float iso, visibleCube_t * cube, int * indexCube, int levelO, int levelC, int nLevel, float maxHeight, int3 realDim, float * r, float * g, float * b, float * screen)
{
	unsigned int tid = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < numRays)
	{
		tid = indexCube[tid];

		if (cube[tid].state == CUDA_NOCUBE)
		{
			screen[tid*3] = r[NUM_COLORS];
			screen[tid*3+1] = g[NUM_COLORS];
			screen[tid*3+2] = b[NUM_COLORS];
			cube[tid].state = CUDA_PAINTED;
			return;
		}
		else if (cube[tid].state == CUDA_CACHED)
		{
			float tnear;
			float tfar;
			// To do test intersection real cube position
			int3 minBox = getMinBoxIndex2(cube[tid].id, levelO, nLevel);
			int dim = powf(2,nLevel-levelO);
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
				int3 minBoxD = getMinBoxIndex2(cube[tid].id >> (3*(levelO - levelC)), levelC, nLevel) - make_int3(CUBE_INC, CUBE_INC, CUBE_INC);
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
							ant = getElementInterpolateGrid(xyz, cube[tid].data, minBoxD, dimD);
							Xfar = Xnear;
							primera = false;
						}
						else
						{
							sig = getElementInterpolateGrid(xyz, cube[tid].data, minBoxD, dimD);

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

					#if 1
					float3 xyz = make_float3(	pos.x + ((Xnear.x - tex1Dfetch(xgrid, pos.x + CUBE_INC)) / (tex1Dfetch(xgrid, pos.x+1 + CUBE_INC) - tex1Dfetch(xgrid, pos.x + CUBE_INC))),
												pos.y + ((Xnear.y - tex1Dfetch(ygrid, pos.y + CUBE_INC)) / (tex1Dfetch(ygrid, pos.y+1 + CUBE_INC) - tex1Dfetch(ygrid, pos.y + CUBE_INC))),
												pos.z + ((Xnear.z - tex1Dfetch(zgrid, pos.z + CUBE_INC)) / (tex1Dfetch(zgrid, pos.z+1 + CUBE_INC) - tex1Dfetch(zgrid, pos.z + CUBE_INC))));
					#else
					float3 xyz = make_float3(	pos.x + ((Xnew.x-xGrid[pos.x])/(xGrid[pos.x+1]-xGrid[pos.x])),
												pos.y + ((Xnew.y-yGrid[pos.y])/(yGrid[pos.y+1]-yGrid[pos.y])),
												pos.z + ((Xnew.z-zGrid[pos.z])/(zGrid[pos.z+1]-zGrid[pos.z])));
					#endif

					float3 n = getNormal(xyz, cube[tid].data, minBoxD,  dimD);
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
					cube[tid].state= CUDA_PAINTED;
				}
				else
				{
					cube[tid].state = CUDA_NOCUBE;
				}
			}
			else
			{
				screen[tid*3] = r[NUM_COLORS];
				screen[tid*3+1] = g[NUM_COLORS];
				screen[tid*3+2] = b[NUM_COLORS];
				cube[tid].state = CUDA_PAINTED;
				return;
			}
		}
	}
}

__global__ void cuda_rayCaster_Cubes(float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int numRays, visibleCube_t * cube, int * indexCube, int levelO, int nLevel, float maxHeight, int3 realDim, float * r, float * g, float * b, float * screen)
{
	unsigned int tid = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < numRays)
	{
		tid = indexCube[tid];

		if (cube[tid].state == CUDA_NOCUBE)
		{
			screen[tid*3] = r[NUM_COLORS];
			screen[tid*3+1] = g[NUM_COLORS];
			screen[tid*3+2] = b[NUM_COLORS];
			cube[tid].state = CUDA_PAINTED;
			return;
		}
		else if (cube[tid].state == CUDA_CUBE)
		{
			int i = tid % pvpW;
			int j = tid / pvpW;

			float3 ray = LB - origin;
			ray += (j*h)*up + (i*w)*right;
			ray = normalize(ray);

			int3 minBoxC = getMinBoxIndex2(cube[tid].id, levelO, nLevel);
			int dim = powf(2,nLevel-levelO);
			int3 maxBoxC = minBoxC + make_int3(dim,dim,dim);

			float3 minBox = _cuda_BoxToCoordinates(minBoxC, realDim);
			float3 maxBox = _cuda_BoxToCoordinates(maxBoxC, realDim);

			float tnear = 0.0f;
			float tfar = 0.0f;
			_cuda_RayAABB(origin, ray,  &tnear, &tfar, minBox, maxBox);
			float3 hit = origin + tnear *ray;

			float3 n = make_float3(0.0f,0.0f,0.0f);
			float aux = 0.0f;

			if (fabsf(maxBox.x - origin.x) < fabsf(minBox.x - origin.x))
			{
				aux = minBox.x;
				minBox.x = maxBox.x; 
				maxBox.x = aux;
			}
			if (fabsf(maxBox.y - origin.y) < fabsf(minBox.y - origin.y))
			{
				aux = minBox.y;
				minBox.y = maxBox.y; 
				maxBox.y = aux;
			}
			if (fabsf(maxBox.z - origin.z) < fabsf(minBox.z - origin.z))
			{
				aux = minBox.z;
				minBox.z = maxBox.z; 
				maxBox.z = aux;
			}

			if(fabsf(hit.x - minBox.x) < EPS) 
				n.x = -1.0f;
			else if(fabsf(hit.x - maxBox.x) < EPS) 
				n.x = 1.0f;
			else if(fabsf(hit.y - minBox.y) < EPS) 
				n.y = -1.0f;
			else if(fabsf(hit.y - maxBox.y) < EPS) 
				n.y = 1.0f;
			else if(fabsf(hit.z - minBox.z) < EPS) 
				n.z = -1.0f;
			else if(fabsf(hit.z - maxBox.z) < EPS) 
				n.z = 1.0f;


			float3 l = hit - origin;// ligth; light on the camera
			l = normalize(l);	
			float dif = fabsf(n.x*l.x + n.y*l.y + n.z*l.z);

			float a = hit.y/maxHeight;
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
			cube[tid].state= CUDA_PAINTED;
		}
	}
}

	void rayCaster(float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int numRays, int levelO, int levelC, int nLevel, float iso, visibleCube_t * cube, int * indexCube, float maxHeight, float * pixelBuffer, int3 realDim, float * r, float * g, float * b, cudaStream_t stream)
{
	dim3 threads = getThreads(numRays);
	dim3 blocks = getBlocks(numRays);

	cuda_rayCaster<<<blocks, threads, 0, stream>>>(origin, LB, up, right, w, h, pvpW, pvpH, numRays, iso, cube, indexCube, levelO, levelC, nLevel, maxHeight, realDim, r, g, b, pixelBuffer);
	#ifndef NDEBUG
	if (cudaSuccess != cudaDeviceSynchronize())
	{
		std::cerr<<"Error ray caster: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	#endif
}
	void rayCasterCubes(float3 origin, float3  LB, float3 up, float3 right, float w, float h, int pvpW, int pvpH, int numRays, int levelO, int nLevel, visibleCube_t * cube, int * indexCube, float maxHeight, float * pixelBuffer, int3 realDim, float * r, float * g, float * b, cudaStream_t stream)
{
	dim3 threads = getThreads(numRays);
	dim3 blocks = getBlocks(numRays);

	cuda_rayCaster_Cubes<<<blocks, threads, 0, stream>>>(origin, LB, up, right, w, h, pvpW, pvpH, numRays, cube, indexCube, levelO, nLevel, maxHeight, realDim, r, g, b, pixelBuffer);
	#ifndef NDEBUG
	if (cudaSuccess != cudaDeviceSynchronize())
	{
		std::cerr<<"Error ray caster: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	#endif
}
}
