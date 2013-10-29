/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlPlaneCache.h>

#include <memoryCheck.h>

#include <cuda_runtime.h>

#ifdef TIMING
#include <lunchbox/clock.h>
#endif

#define READING -1
#define WAITING 200

namespace eqMivt
{

bool ControlPlaneCache::initParameter(std::vector<std::string> file_parameters, float memoryOccupancy)
{
	if (memoryOccupancy <= 0.0f || memoryOccupancy > 1.0f)
	{
		std::cerr<<"Control Plane Cache, Memory occupancy may be > 0.0 and <= 1.0f "<<memoryOccupancy<<std::endl;
		return false;
	}

	_memoryOccupancy = memoryOccupancy;
	_file_parameters = file_parameters;

	return  ControlElementCache::_init();
}

bool ControlPlaneCache::_threadInit()
{
	_min = vmml::vector<3,int>(0,0,0);
	_max = vmml::vector<3,int>(0,0,0);
	_minFuture = vmml::vector<3,int>(-1,-1,-1);
	_maxFuture = vmml::vector<3,int>(-1,-1,1);

	_memoryAviable = 0;

	_maxNumPlanes = 0;

	return _file.init(_file_parameters) &&  ControlElementCache::_threadInit();
}

void ControlPlaneCache::_threadStop()
{
	if (_memory != 0)
		if(cudaSuccess != cudaFreeHost((void*)_memory))
		{
			std::cerr<<"Control Plane Cache, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}
	_file.close();

	ControlElementCache::_threadStop();
}

void ControlPlaneCache::_freeCache()
{
	#ifdef TIMING
	std::cout<<"==== CONTROL PLANE CACHE ====="<<std::endl;
	std::cout<<"Max num planes "<< _maxNumPlanes<<" planes form "<<_min.x()<<" to "<<_max.x()<<" total "<<_max.x()-_min.x()<<std::endl;
	#endif

	ControlElementCache::_freeCache();

	#ifdef TIMING
	std::cout<<"=============================="<<std::endl;
	#endif
}


void ControlPlaneCache::_reSizeCache()
{
	// If the same new paramenters avoid resize
	if (_min == _minFuture && _max == _maxFuture)
		return;

	vmml::vector<3, int> dimVolume = _file.getRealDimension();

	if (_minFuture.x() >= _maxFuture.x() || _minFuture.y() >= _maxFuture.y() || _minFuture.z() >= _maxFuture.z())
	{
		std::cerr<<"Control Plane Cache, minimum and maximum coordinates should be min < max"<<std::endl;
		throw;
	}
	if (_minFuture.x() < 0 || _minFuture.y()  < 0 || _minFuture.z() < 0)
	{
		std::cerr<<"Control Plane Cache, minimum coordinates should be >= 0"<<std::endl;
		throw;
	}
	if (_maxFuture.x() > dimVolume.x() || _maxFuture.y() > dimVolume.y() || _maxFuture.z() > dimVolume.z())
	{
		std::cerr<<"Control Plane Cache, maximum coordinates should be <= volume dimension"<<std::endl;
		throw;
	}

	_min = _minFuture;
	_max = _maxFuture;
	_minFuture = vmml::vector<3,int>(0,0,0);
	_maxFuture = vmml::vector<3,int>(0,0,0);

	_sizeElement = (_max.y()-_min.y())*(_max.z()-_min.z());	

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
			_memoryAviable *=0.7*_memoryOccupancy;
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

	_maxNumPlanes = _memoryAviable / (_sizeElement * sizeof(float));
	if (_max.x()-_min.x() + 1 < _maxNumPlanes)
	{
		_maxNumPlanes = _max.x()-_min.x() + 1;
	}
	else
	{
		while((_maxNumPlanes*_sizeElement*sizeof(float) + _maxNumPlanes*sizeof(NodePlane_t)) > _memoryAviable)
		{
			_maxNumPlanes -= 10;
		}
	}

	_freeSlots = _maxNumPlanes;

	ControlElementCache::_reSizeCache();
}

bool ControlPlaneCache::_readElement(NodeLinkedList<int> * element)
{
	float * data = _memory + element->element * _sizeElement;
	int plane = element->id;

    return _file.readPlane(data, vmml::vector<3, int>(plane, _min.y(), _min.z()), vmml::vector<3, int>(plane, _max.y(), _max.z()));
}

bool ControlPlaneCache::freeCacheAndPause()
{
	return ControlElementCache::_freeCacheAndPause();
}

bool ControlPlaneCache::reSizeCacheAndContinue(vmml::vector<3,int> min, vmml::vector<3,int> max)
{
	_minFuture = min;
	_maxFuture = max;

	return ControlElementCache::_reSizeCacheAndContinue();
}

}
