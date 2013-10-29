/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlCubeCPUCache.h>

#include <mortonCodeUtil_CPU.h>
#include <memoryCheck.h>

#include <cuda_runtime.h>

namespace eqMivt
{

bool ControlCubeCPUCache::initParameter(std::vector<std::string> file_parameters, float memoryOccupancy)
{
	if (memoryOccupancy <= 0.0f || memoryOccupancy > 1.0f)
	{
		std::cerr<<"Control Plane Cache, Memory occupancy may be > 0.0 and <= 1.0f "<<memoryOccupancy<<std::endl;
		return false;
	}
	
	_memoryOccupancy = memoryOccupancy;

	return  _planeCache.initParameter(file_parameters, memoryOccupancy) && ControlElementCache::_init();
}

bool ControlCubeCPUCache::_threadInit()
{
	_offset.set(0,0,0);
	_nextOffset.set(-1,-1,-1);
	_nextnLevels = -1;
	_nextLevelCube = -1;
	_nLevels = 0;
	_levelCube = 0;
	_dimCube = 0;
	_memoryAviable = 0;
	_maxNumCubes = 0;

	return ControlElementCache::_threadInit();
}

void ControlCubeCPUCache::_threadStop()
{
	if (_memory != 0)
		if(cudaSuccess != cudaFreeHost((void*)_memory))
		{
			std::cerr<<"Control Cube CPU Cache, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}

	ControlElementCache::_threadStop();
}

void ControlCubeCPUCache::_freeCache()
{
	#ifdef TIMING
	int dim = exp2(_nLevels);
	index_node_t s = coordinateToIndex(vmml::vector<3, int>(0,0,0), _levelCube, _nLevels);
	index_node_t e = coordinateToIndex(vmml::vector<3, int>(dim-1, dim-1, dim-1), _levelCube, _nLevels);
	std::cout<<"==== CONTROL CUBE CPU CACHE ====="<<std::endl;
	std::cout<<"Max num cubes "<< _maxNumCubes<<" cubes form "<<s<<" to "<<e<<" total "<<e-s+1<<std::endl;
	#endif

	ControlElementCache::_freeCache();

	#ifdef TIMING
	std::cout<<"=============================="<<std::endl;
	#endif
}

void ControlCubeCPUCache::_reSizeCache()
{
	_nLevels = _nextnLevels;
	_levelCube = _nextLevelCube;
	_offset	= _nextOffset;
	_nextnLevels = 0;
	_nextLevelCube = 0;

	_dimCube = exp2(_nLevels - _levelCube) + 2 * CUBE_INC;

	_sizeElement = pow(_dimCube, 3); 

	if (_memory == 0)
	{
		_memoryAviable = getMemorySize();
		if (_memoryAviable == 0)
		{
			std::cerr<<"Not possible, check memory aviable (the call failed due to OS limitations)"<<std::endl;
			_memoryAviable = 1024*1024*1024;
		}
		else
		{
			_memoryAviable *=MEMORY_OCCUPANCY_CUBE_CACHE*_memoryOccupancy;
		}
		while (cudaSuccess != cudaHostAlloc((void**)&_memory, _memoryAviable, cudaHostAllocDefault))
		{                                                                                               
			std::cerr<<"Control Plane Cache, error allocating memory "<<_memoryAviable/1024.0f/1024.0f<<" : "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			if (_memoryAviable <= 0)
				throw;
			else
				_memoryAviable -= _memoryAviable * 0.1;
		}

	}

	_maxNumCubes = _memoryAviable / (_sizeElement * sizeof(float));

	_freeSlots = _maxNumCubes;

	ControlElementCache::_reSizeCache();
}

bool ControlCubeCPUCache::_readElement(NodeLinkedList<index_node_t> * element)
{
	vmml::vector<3, int> coordS = getMinBoxIndex2(element->id, _levelCube, _nLevels) + _offset - vmml::vector<3, int>(CUBE_INC, CUBE_INC, CUBE_INC);
	vmml::vector<3, int> coordE = coordS + vmml::vector<3, int>(_dimCube, _dimCube, _dimCube);

	vmml::vector<2, int> planeDim = _planeCache.getPlaneDim();
	vmml::vector<3, int> minC = _planeCache.getMinCoord();
	vmml::vector<3, int> maxC = _planeCache.getMaxCoord();

	std::vector<int>::iterator it = element->pendingPlanes.begin(); 
	while (it != element->pendingPlanes.end())
	{
		float * start = (_memory + element->element*_sizeElement) + _dimCube * _dimCube * abs(coordS.x() - (*it));
		float * plane = _planeCache.getAndBlockElement(*it);

		if (plane != 0)
		{
			start += coordS.z() < 0 ? abs(coordS.z()) : 0;
			plane += coordS.z() <= minC.z() ? 0 : coordS.z() - minC.z();
			int dimC = _dimCube;

			dimC -= coordS.z() < minC.z() ? abs(minC.z() - coordS.z()) : 0;
			dimC -= coordE.z() > maxC.z() ? coordE.z() - maxC.z() : 0;

			int sL = coordS.y() < minC.y() ? abs(minC.y() - coordS.y()) : 0;
			int lL = coordE.y() > maxC.y() ? _dimCube - (coordE.y() - maxC.y()) : _dimCube;

			for(int i = sL; i < lL; i++)
			{
				int planeI = i+(coordS.y() - minC.y());
				memcpy((void*)(start + i*_dimCube), (void*)(plane + planeI*planeDim.y()), dimC*sizeof(float));
			}

			_planeCache.unlockElement(*it);
			it = element->pendingPlanes.erase(it);
		}

		if (it != element->pendingPlanes.end())
			it++;
	}

	return element->pendingPlanes.empty();
}

bool ControlCubeCPUCache::freeCacheAndPause()
{
	return _planeCache.freeCacheAndPause() && ControlElementCache::_freeCacheAndPause();
}

bool ControlCubeCPUCache::reSizeCacheAndContinue(vmml::vector<3,int> offset, vmml::vector<3,int> max, int levelCube, int nLevels)
{
	vmml::vector<3,int> min;
	min[0] = offset.x() - CUBE_INC < 0 ? 0 : offset.x() - CUBE_INC;
	min[1] = offset.y() - CUBE_INC < 0 ? 0 : offset.y() - CUBE_INC;
	min[2] = offset.z() - CUBE_INC < 0 ? 0 : offset.z() - CUBE_INC;

	_nextOffset	= offset;
	_nextnLevels = nLevels;
	_nextLevelCube = levelCube;

	return _planeCache.reSizeCacheAndContinue(min, max) && ControlElementCache::_reSizeCacheAndContinue();
}

}
