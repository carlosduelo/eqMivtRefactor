/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/


#include <octree.h>

#include <octree_cuda.h>

namespace eqMivt
{

Octree::Octree()
{
	_oc = 0;

	_xGrid = 0;
	_yGrid = 0;
	_zGrid = 0;

	_currentOctree = -1;
	_memoryOctree = 0;
	_octree = 0;
	_sizes = 0;

	_device = 0;
}

bool Octree::init(OctreeContainer * oc, device_t device)
{
	_oc = oc;
	_device = device;
	vmml::vector<3, int> dim = oc->getRealDimVolume();

	if (cudaSuccess != cudaSetDevice(_device))
	{
		std::cerr<<"Octree, cudaSetDevice error: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}

	if (cudaSuccess != cudaMalloc((void**)&_xGrid, (dim.x()+4)*sizeof(float)) ||
		cudaSuccess != cudaMalloc((void**)&_yGrid, (dim.y()+4)*sizeof(float)) ||
		cudaSuccess != cudaMalloc((void**)&_zGrid, (dim.z()+4)*sizeof(float)))
	{
		std::cerr<<"Octree, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}

	if (cudaSuccess != cudaMemcpy((void*)_xGrid, (void*)oc->getXGrid(), (dim.x()+4)*sizeof(float), cudaMemcpyHostToDevice) ||
		cudaSuccess != cudaMemcpy((void*)_yGrid, (void*)oc->getYGrid(), (dim.y()+4)*sizeof(float), cudaMemcpyHostToDevice) ||
		cudaSuccess != cudaMemcpy((void*)_zGrid, (void*)oc->getZGrid(), (dim.z()+4)*sizeof(float), cudaMemcpyHostToDevice))
	{
		std::cerr<<"Octree, error copying grid: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}

	return true;
}

bool Octree::loadCurrentOctree()
{
	if (_currentOctree == _oc->getCurrentOctree())
		return true;
		
	if (cudaSuccess != cudaSetDevice(_device))
	{
		std::cerr<<"Octree, cudaSetDevice error: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}

	if (_memoryOctree != 0)
		if (cudaSuccess != cudaFree(_memoryOctree))
		{
			std::cerr<<"Octree, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return false;
		}
	if (_sizes != 0)
		if (cudaSuccess != cudaFree(_sizes))
		{
			std::cerr<<"Octree, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return false;
		}
	if (_octree != 0)
		if (cudaSuccess != cudaFree(_octree))
		{
			std::cerr<<"Octree, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return false;
		}

	int l = _oc->getmaxLevel() + 1;

	int  s = 0;
	for(int i=0; i<l; i++)
	{
		s += _oc->getSizes()[i];
	}

	if (cudaSuccess != cudaMalloc((void**)&_memoryOctree, s*sizeof(index_node_t)) ||
		cudaSuccess != cudaMalloc((void**)&_sizes, l*sizeof(int)) ||
		cudaSuccess != cudaMalloc((void**)&_octree, l*sizeof(index_node_t*)))
	{
		std::cerr<<"Octree, allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}

	if (cudaSuccess != cudaMemcpy((void*)_memoryOctree, (void*)_oc->getOctree(), s*sizeof(index_node_t), cudaMemcpyHostToDevice) || 
		cudaSuccess != cudaMemcpy((void*)_sizes, (void*)_oc->getSizes(), l*sizeof(int), cudaMemcpyHostToDevice)) 
		{
			std::cerr<<"Octree, copying octree: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return false;
		}
	
	insertOctreePointers(_octree, _sizes, _memoryOctree, l);

	return true;
}

void Octree::stop()
{
	if (cudaSuccess != cudaSetDevice(_device))
	{
		std::cerr<<"Octree, cudaSetDevice error: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return;
	}

	if (_xGrid != 0)
		if (cudaSuccess != cudaFree(_xGrid))
		{
			std::cerr<<"Octree, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return;
		}
	if (_yGrid != 0)
		if (cudaSuccess != cudaFree(_yGrid))
		{
			std::cerr<<"Octree, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return;
		}
	if (_zGrid != 0)
		if (cudaSuccess != cudaFree(_zGrid))
		{
			std::cerr<<"Octree, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return;
		}

	if (_memoryOctree != 0)
		if (cudaSuccess != cudaFree(_memoryOctree))
		{
			std::cerr<<"Octree, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return;
		}
	if (_sizes != 0)
		if (cudaSuccess != cudaFree(_sizes))
		{
			std::cerr<<"Octree, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return;
		}
	if (_octree != 0)
		if (cudaSuccess != cudaFree(_octree))
		{
			std::cerr<<"Octree, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return;
		}
}

float * Octree::getxGrid()
{
	return &_xGrid[CUBE_INC + _oc->getStartCoord().x()];
}

float * Octree::getyGrid()
{
	return &_yGrid[CUBE_INC + _oc->getStartCoord().y()];
}
float * Octree::getzGrid()
{
	return &_zGrid[CUBE_INC + _oc->getStartCoord().z()];
}
}
