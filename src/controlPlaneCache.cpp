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
		return false;
}

bool ControlPlaneCache::initParameter(std::vector<std::string> file_parameters)
{
	bool result = _file.init(file_parameters);

	_min = vmml::vector<3,int>(0,0,0);
	_max = vmml::vector<3,int>(0,0,0);
	_minFuture = vmml::vector<3,int>(0,0,0);
	_maxFuture = vmml::vector<3,int>(0,0,0);

	_memoryAviable = 0;
	_memoryPlane = 0;
	_cachePlanes = 0;
	_lruPlanes.clear();
	_currentPlanes.clear();
	_pendingPlanes.clear();

	_freeSlots = 0;
	_maxNumPlanes = 0;
	_sizePlane = 0;

	_end = false;
	_resize = false;

	return result;;
}

void ControlPlaneCache::reSizeStructures()
{
	vmml::vector<3, int> dimVolume = _file.getRealDimension();

	if (_minFuture.x() >= _maxFuture.x() || _minFuture.y() >= _maxFuture.y() || _minFuture.z() >= _maxFuture.z())
	{
		std::cerr<<"Control Plane Cache, minimum and maximum coordinates should be min > max"<<std::endl;
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

	_sizePlane = (_max.y()-_min.y())*(_max.z()-_min.z());	

	if (_cachePlanes != 0)
		delete[] _cachePlanes;
	_cachePlanes = 0;

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
			_memoryAviable *=0.7;
		}
		if (cudaSuccess != cudaHostAlloc((void**)&_memoryPlane, _memoryAviable, cudaHostAllocDefault))
		{                                                                                               
			std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}

	}

	_maxNumPlanes = _memoryAviable / (_sizePlane * sizeof(float));
	while((_maxNumPlanes*_sizePlane*sizeof(float) + _maxNumPlanes*sizeof(cache_plane_t)) > _memoryAviable)
	{
		_maxNumPlanes -= 10;
	}

	_freeSlots = _maxNumPlanes;

	_cachePlanes = new cache_plane_t[_maxNumPlanes];
	if (_cachePlanes == 0)
	{
		std::cerr<<"ControlPlaneCache, error creating"<<std::endl;
		throw;
	}

	_lruPlanes.clear();
	_currentPlanes.clear();
	_pendingPlanes.clear();

	for(int i=0; i<_maxNumPlanes; i++)
	{
		_cachePlanes[i].id = -1;
		_cachePlanes[i].data = _memoryPlane + i*_sizePlane;
		_cachePlanes[i].refs = 0;
		_cachePlanes[i].timestamp = std::time(0);
		_lruPlanes.push_back(&_cachePlanes[i]);
	}

	_end = false;
	_resize = false;
}

bool ControlPlaneCache::reSize(vmml::vector<3,int> min, vmml::vector<3,int> max)
{
	_lockResize.lock();

		_emptyPendingPlanes.signal();
		_fullSlots.signal();
		_resize = true;

		_minFuture = min;
		_maxFuture = max;

	_lockResize.wait();

	_lockResize.unlock();

	return true;
}

vmml::vector<2,int>		ControlPlaneCache::getPlaneDim()
{
	_lockResize.lock();
	if (_resize)
		_lockResize.wait();

	vmml::vector<2,int> r (_max.y() - _min.y(), _max.z() - _min.z()); 
	_lockResize.unlock();

	return r;
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
	_lockResize.lock();
	if (_resize)
		_lockResize.wait();

	if (_memoryPlane != 0)
		if(cudaSuccess != cudaFreeHost((void*)_memoryPlane))
		{
			std::cerr<<"Control Plane Cache, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}
	if (_cachePlanes != 0)
		delete[] _cachePlanes;

	_lockResize.unlock();
}

float * ControlPlaneCache::getAndBlockPlane(int plane)
{
	_lockResize.lock();
		if (_resize)
		{
			_lockResize.wait();
			_lockResize.unlock();
			return 0;
		}
	_lockResize.unlock();

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
	_lockResize.lock();
		if (_resize)
		{
			_lockResize.wait();
			_lockResize.unlock();
		}
	_lockResize.unlock();

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

			_lockResize.lock();
				if (_resize)
				{
					reSizeStructures();	
					_lockResize.broadcast();
					_lockResize.unlock();
					_emptyPendingPlanes.unlock();
					continue;
				}
			_lockResize.unlock();
				
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

			_lockResize.lock();
				if (_resize)
				{
					reSizeStructures();	
					_lockResize.broadcast();
					_lockResize.unlock();
					_fullSlots.unlock();
					continue;
				}
			_lockResize.unlock();

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
