/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlPlaneCache.h>

#include <memoryCheck.h>

#include <cuda_runtime.h>

#define READING -1

namespace eqMivt
{

bool compareCachePlane(cache_plane_t * a, cache_plane_t * b)
{
	if (a->refs > 0 && b->refs > 0)
		return a->timestamp < b->timestamp;
	else if (a->refs == READING)
		return false;
	else if (b->refs == READING)
		return true;
	else if (a->refs == 0 && b->refs == 0)
		return a->timestamp < b->timestamp;
	else
		return false;//a->refs == 0;
}

bool ControlPlaneCache::initParameter(std::vector<std::string> file_parameters, vmml::vector<3, int> min, vmml::vector<3, int> max)
{
	_file.init(file_parameters);

	vmml::vector<3, int> dimVolume = _file.getRealDimension();

	if (min == vmml::vector<3, int>(0,0,0) && max == vmml::vector<3, int>(0,0,0))
	{
		_min = vmml::vector<3, int>(0,0,0);
		_max = dimVolume;
	}
	else
	{
		if (min.x() >= max.x() || min.y() >= max.y() || min.z() >= max.z())
		{
			std::cerr<<"Control Plane Cache, minimum and maximum coordinates should be min > max"<<std::endl;
			return false;
		}
		if (min.x() < 0 || min.y()  < 0 || min.z() < 0)
		{
			std::cerr<<"Control Plane Cache, minimum coordinates should be >= 0"<<std::endl;
			return false;
		}
		if (max.x() > dimVolume.x() || max.y() > dimVolume.y() || max.z() > dimVolume.z())
		{
			std::cerr<<"Control Plane Cache, maximum coordinates should be <= volume dimension"<<std::endl;
			return false;
		}
	
		_min = min;
		_max = max;
	}

	_sizePlane = (max.x()-min.x())*(max.y()-min.y())*(max.z()-min.z());	

	double memory = getMemorySize();
	if (memory == 0)
	{
		std::cerr<<"Not possible, check memory aviable (the call failed due to OS limitations)"<<std::endl;
		memory = 1024*1024*1024;
	}
	else
	{
		memory *=0.6;
	}

	_maxNumPlanes = memory / (_sizePlane * sizeof(float));
	while((_maxNumPlanes*_sizePlane*sizeof(float) + _maxNumPlanes*sizeof(cache_plane_t)) > memory)
	{
		_maxNumPlanes -= 10;
	}
	
	_freeSlots = _maxNumPlanes;

	if (cudaSuccess != cudaHostAlloc((void**)&_memoryPlane, _sizePlane*_maxNumPlanes*sizeof(float), cudaHostAllocDefault))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}

	_cachePlanes = 0;
	_cachePlanes = new cache_plane_t[_maxNumPlanes];
	if (_cachePlanes == 0)
	{
		std::cerr<<"ControlPlaneCache, error creating"<<std::endl;
		throw;
	}

	for(int i=0; i<_maxNumPlanes; i++)
	{
		_cachePlanes[i].id = -1;
		_cachePlanes[i].data = _memoryPlane + i*_sizePlane;
		_cachePlanes[i].refs = 0;
		_cachePlanes[i].timestamp = std::time(0);
		_lruPlanes.push_back(&_cachePlanes[i]);
	}

	_end = false;

	return true;
}

vmml::vector<2,int>		ControlPlaneCache::getPlaneDim()
{
	return vmml::vector<2,int>(_max.y() - _min.y(), _max.z() - _min.z()); 
}

void ControlPlaneCache::stopProcessing()
{
	_lockEnd.set();
		_emptyPendingPlanes.broadcast();
		_fullSlots.broadcast();
		_end = true;
	_lockEnd.unset();

	join();
}

ControlPlaneCache::~ControlPlaneCache()
{
	if (_memoryPlane != 0)
		if(cudaSuccess != cudaFreeHost((void*)_memoryPlane))
		{
			std::cerr<<"Control Plane Cache, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}
	if (_cachePlanes != 0)
		delete[] _cachePlanes;	
}

float * ControlPlaneCache::getAndBlockPlane(int plane)
{
	#ifndef NDEBUG
	if (plane >= _max.x() || plane < _min.x())
	{
		std::cerr<<"Control Plane Cache, error adding plane that not exists"<<std::endl;
		return 0;
	}

	#endif
	float * dplane = 0;
	boost::unordered_map<int, cache_plane_t * >::iterator it;

	_fullSlots.lock();

	it = _currentPlanes.find(plane);
	if (it != _currentPlanes.end())
	{
		if (it->second->refs == 0)
			_freeSlots--;
		else if (it->second->refs == READING)
			it->second->refs = 0;

		it->second->refs += 1;
		it->second->timestamp = std::time(0); 
		dplane = it->second->data;

		std::sort(_lruPlanes.begin(), _lruPlanes.end(), compareCachePlane);
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
	boost::unordered_map<int, cache_plane_t *>::iterator it;

	_fullSlots.lock();

	it = _currentPlanes.find(plane);
	if (it != _currentPlanes.end())
	{
		it->second->refs -= 1;
		if (it->second->refs == 0)
		{
			_freeSlots++;
			_fullSlots.signal();
		}
		
		std::sort(_lruPlanes.begin(), _lruPlanes.end(), compareCachePlane);

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

	_fullSlots.unlock();
}

bool ControlPlaneCache::readPlane(float * data, int plane)
{
	_file.readPlane(data, vmml::vector<3, int>(plane, _min.y(), _min.z()), vmml::vector<3, int>(plane, _max.y(), _max.z()));

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
				exit();
			}
			_lockEnd.unset();

			int plane = _pendingPlanes.front();
			_pendingPlanes.erase(_pendingPlanes.begin());
		_emptyPendingPlanes.unlock();

		_fullSlots.lock();
			if (_freeSlots == 0)
				_fullSlots.wait();

			_lockEnd.set();
			if (_end)
			{
				_lockEnd.unset();
				_fullSlots.unlock();
				exit();
			}
			_lockEnd.unset();

			cache_plane_t * c = _lruPlanes.front();

			boost::unordered_map<int, cache_plane_t *>::iterator it;
			it = _currentPlanes.find(c->id);
			if (it != _currentPlanes.end())
				_currentPlanes.erase(it);	

			#ifndef NDEBUG
			if (c->refs != 0)
			{
				std::cerr<<"Control Plane Cache, unistable state, free plane slot with references "<<c->id<<" refs "<<c->refs<<std::endl;
				while(_lruPlanes.size() != 0)
				{
					cache_plane_t * p = _lruPlanes[0];
					_lruPlanes.erase(_lruPlanes.begin());
					std::cerr<<"Plane "<<p->id<<" "<<p->refs<<std::endl;
				}
				throw;
			}
			#endif

			it = _currentPlanes.find(plane);
			if (it == _currentPlanes.end())
			{
				_freeSlots--;
				
				c->id = plane;
				c->refs = READING;

				readPlane(c->data, c->id);
				_currentPlanes.insert(std::make_pair<int, cache_plane_t *>(c->id, c));

				std::sort(_lruPlanes.begin(), _lruPlanes.end(), compareCachePlane);
			}

		_fullSlots.unlock();
	}
	
}

}

