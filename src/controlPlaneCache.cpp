/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlPlaneCache.h>

#include <memoryCheck.h>

#include <cuda_runtime.h>

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

	return  ControlCache::_initControlCache();
}

bool ControlPlaneCache::_threadInit()
{
	_min = vmml::vector<3,int>(0,0,0);
	_max = vmml::vector<3,int>(0,0,0);
	_minFuture = vmml::vector<3,int>(-1,-1,-1);
	_maxFuture = vmml::vector<3,int>(-1,-1,1);

	_memoryAviable = 0;
	_memoryPlane = 0;
	_currentPlanes.clear();
	_pendingPlanes.clear();

	_freeSlots = 0;
	_maxNumPlanes = 0;
	_sizePlane = 0;

	_lastPlane = 0;

	return _file.init(_file_parameters);
}

void ControlPlaneCache::_threadStop()
{
	if (_memoryPlane != 0)
		if(cudaSuccess != cudaFreeHost((void*)_memoryPlane))
		{
			std::cerr<<"Control Plane Cache, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}
	_memoryPlane = 0;
	_file.close();
}

void ControlPlaneCache::_threadWork()
{
	bool prefetching = false;
	int plane = -1;

	_emptyPendingPlanes.lock();

		if (_pendingPlanes.size() == 0)
		{
			if (!_emptyPendingPlanes.timedWait(WAITING))
				prefetching = true;
		}
		if (prefetching)
			plane = (_lastPlane + 1) < _max.x() ? _lastPlane + 1 : _min.x();
		else
			plane = _pendingPlanes.front();

	_emptyPendingPlanes.unlock();

	_fullSlots.lock();

		if (_freeSlots == 0)
		{
			if (!_fullSlots.timedWait(WAITING))
			{
				_fullSlots.unlock();
				return;
			}
		}
		boost::unordered_map<int, NodeLinkedList *>::iterator it;
		it = _currentPlanes.find(plane);
		if (it == _currentPlanes.end())
		{
			NodeLinkedList * c = 0;
			c = _lruPlanes.getFirstFreePosition();

			it = _currentPlanes.find(c->id);
			if (it != _currentPlanes.end())
				_currentPlanes.erase(it);	

			#ifndef NDEBUG
			if (c->refs != 0)
			{
				std::cerr<<"Control Plane Cache, unistable state, free plane slot with references "<<c->id<<" refs "<<c->refs<<std::endl;
				throw;
			}
			#endif

			if (readPlane(_memoryPlane + (c->element * _sizePlane), plane))
			{
				_freeSlots--;
				
				c->id = plane;
				c->refs = READING;
				_currentPlanes.insert(std::make_pair<int, NodeLinkedList *>(c->id, c));
				_lruPlanes.moveToLastPosition(c);
				_lastPlane = c->id;

				if (!prefetching)
				{
					_emptyPendingPlanes.lock();
						_pendingPlanes.erase(_pendingPlanes.begin());
					_emptyPendingPlanes.unlock();
				}
			}
		}

	_fullSlots.unlock();

	return;
}

void ControlPlaneCache::_freeCache()
{
	// DO NOT FREE, FREE AT THE END 
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
	_lastPlane = _min.x();

	_sizePlane = (_max.y()-_min.y())*(_max.z()-_min.z());	

	if (_memoryPlane == 0)
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
		while (cudaSuccess != cudaHostAlloc((void**)&_memoryPlane, _memoryAviable, cudaHostAllocDefault))
		{                                                                                               
			std::cerr<<"Control Plane Cache, error allocating memory "<<_memoryAviable/1024.0f/1024.0f<<" : "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			if (_memoryAviable <= 0)
				throw;
			else
				_memoryAviable -= _memoryAviable * 0.1;
		}

	}

	_maxNumPlanes = _memoryAviable / (_sizePlane * sizeof(float));
	if (_max.x()-_min.x() + 1 < _maxNumPlanes)
	{
		_maxNumPlanes = _max.x()-_min.x() + 1;
	}
	else
	{
		while((_maxNumPlanes*_sizePlane*sizeof(float) + _maxNumPlanes*sizeof(NodeLinkedList)) > _memoryAviable)
		{
			_maxNumPlanes -= 10;
		}
	}

	_freeSlots = _maxNumPlanes;

	_lruPlanes.reSize(_maxNumPlanes);
	_currentPlanes.clear();
	_pendingPlanes.clear();
}

bool ControlPlaneCache::readPlane(float * data, int plane)
{
    return _file.readPlane(data, vmml::vector<3, int>(plane, _min.y(), _min.z()), vmml::vector<3, int>(plane, _max.y(), _max.z()));
}

bool ControlPlaneCache::freeCacheAndPause()
{
	return _pauseWorkAndFreeCache();
}

bool ControlPlaneCache::reSizeCacheAndContinue(vmml::vector<3,int> min, vmml::vector<3,int> max)
{
	if (_checkRunning())
		return false;

	_minFuture = min;
	_maxFuture = max;

	return _continueWorkAndReSize();
}

float * ControlPlaneCache::getAndBlockPlane(int plane)
{
	#ifndef NDEBUG
	if (plane >= _max.x() || plane < _min.x())
	{
		std::cerr<<"Control Plane Cache, error adding plane that not exists "<<_min<<" "<<_max<<" "<<plane<<std::endl;
		return 0;
	}
	#endif

	float * dplane = 0;
	boost::unordered_map<int, NodeLinkedList * >::iterator it;

	_fullSlots.lock();

	it = _currentPlanes.find(plane);
	if (it != _currentPlanes.end())
	{
		if (it->second->refs == 0)
			_freeSlots--;
		else if (it->second->refs == READING)
			it->second->refs = 0;

		it->second->refs += 1;
		dplane = _memoryPlane + (it->second->element * _sizePlane);

		_lruPlanes.moveToLastPosition(it->second);
	}
	else
	{
		_emptyPendingPlanes.lock();
			if (std::find(_pendingPlanes.begin(), _pendingPlanes.end(), plane) == _pendingPlanes.end())
				_pendingPlanes.push_back(plane);
			if (_pendingPlanes.size() == 1)
				_emptyPendingPlanes.signal();
		_emptyPendingPlanes.unlock();
	}

	_fullSlots.unlock();

	return dplane;
}

void	ControlPlaneCache::unlockPlane(int plane)
{
	boost::unordered_map<int, NodeLinkedList *>::iterator it;

	_fullSlots.lock();

	it = _currentPlanes.find(plane);
	if (it != _currentPlanes.end())
	{
		it->second->refs -= 1;

		#ifndef NDEBUG
		if (it->second->refs < 0)
		{
			std::cerr<<"Control Plane Cache, error unlocking plane"<<std::endl;
			throw;
		}
		#endif

		if (it->second->refs == 0)
		{
			_freeSlots++;
			_fullSlots.signal();
		}

	}
	#ifndef NDEBUG
	else
	{
		std::cerr<<"Control Plane Cache, error unlocking plane that not exists "<<plane<<std::endl;
		throw;
	}
	#endif

	_fullSlots.unlock();
}


}
