/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlPlaneCache.h>

#include <cuda_runtime.h>

namespace eqMivt
{

bool ControlPlaneCache::init()
{
	_maxPlane = 33;

	_sizePlane = 100;	
	_maxNumPlanes = 10;

	if (cudaSuccess != cudaHostAlloc((void**)&_memoryPlane, _sizePlane*_maxNumPlanes*sizeof(float), cudaHostAllocDefault))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory"<<std::endl;
		throw;
	}

	for(int i=0; i<_maxNumPlanes; i++)
	{
		cache_plane_t c;
		c.id = -1;
		c.data = _memoryPlane + i*_sizePlane;
		c.refs = 0;
		c.timestamp = std::time(0);
		_lruPlanes.push(c);
	}

	_end = false;

	return true;
}

void ControlPlaneCache::stopProcessing()
{
	_lockEnd.set();
		_emptyPendingPlanes.broadcast();
		_end = true;
	_lockEnd.unset();

	_lock.set();
		if (_memoryPlane != 0)
			if(cudaSuccess != cudaFreeHost((void*)_memoryPlane))
			{
				std::cerr<<"Control Plane Cache, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
				throw;
			}
		_memoryPlane = 0;
	_lock.unset();
}

void ControlPlaneCache::addPlane(int plane)
{
	#ifndef NDEBUG
	if (plane > _maxPlane)
	{
		std::cerr<<"Control Plane Cache, error adding plane that not exists"<<std::endl;
	}
	#endif

	boost::unordered_map<int, cache_plane_t *>::iterator it;

	_lock.set();

	std::cout<<"ADDING PLANE"<<std::endl;

	it = _currentPlanes.find(plane);
	if (it == _currentPlanes.end())
	{
		_emptyPendingPlanes.lock();
			_pendingPlanes.push(plane);
		_emptyPendingPlanes.signal();
		_emptyPendingPlanes.unlock();
	}
	else
	{
		it->second->timestamp = std::time(0); 
	}

	_lock.unset();
}

void ControlPlaneCache::addPlanes(std::vector<int> planes)
{
	for(std::vector<int>::iterator it = planes.begin(); it!=planes.end(); ++it)
		addPlane(*it);
}

float * ControlPlaneCache::getAndBlockPlane(int plane)
{
	float * dplane = 0;
	boost::unordered_map<int, cache_plane_t *>::iterator it;

	_lock.set();

	it = _currentPlanes.find(plane);
	if (it != _currentPlanes.end())
	{
		it->second->refs += 1;
		dplane = it->second->data;
	}

	_lock.unset();

	return dplane;
}

void	ControlPlaneCache::unlockPlane(int plane)
{
	boost::unordered_map<int, cache_plane_t *>::iterator it;

	_lock.set();

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
	}
	#ifndef NDEBUG
	else
	{
		std::cerr<<"Control Plane Cache, error unlocking plane that not exists"<<std::endl;
		throw;
	}
	#endif

	_lock.unset();
}

bool ControlPlaneCache::readPlane(float * data, int plane)
{
	return true;
}

void ControlPlaneCache::run()
{
	while(1)
	{
		_emptyPendingPlanes.lock();
			if (_pendingPlanes.size() == 0)
				_emptyPendingPlanes.wait();
				
			_lockEnd.set();
			if (_end)
			{
				_lockEnd.unset();
				_emptyPendingPlanes.unlock();
				break;
			}

			int plane = _pendingPlanes.front();
			_pendingPlanes.pop();
			std::cout<<"Processing plane "<<plane<<std::endl;
		_emptyPendingPlanes.unlock();

		_lock.set();
			_lockEnd.unset();
			cache_plane_t c = _lruPlanes.top();
			_lruPlanes.pop();
			_lruPlanes.push(c);
			std::cout<<"Plane processed "<<plane<<std::endl;
		_lock.unset();
	}
}

}

