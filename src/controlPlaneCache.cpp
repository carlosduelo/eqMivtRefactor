/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlPlaneCache.h>

#include <cuda_runtime.h>

namespace eqMivt
{

bool compareCachePlane(cache_plane_t * a, cache_plane_t * b)
{
	if (a->refs > 0 && b->refs > 0)
		return a->timestamp < b->timestamp;
	else if (a->refs == -1)
		return false;
	else if (b->refs == -1)
		return true;
	else
		return a->refs == 0;
}

bool ControlPlaneCache::initParamenter(std::vector<std::string> file_parameters, int maxHeight)
{
	_file.init(file_parameters);
	
	_maxPlane = _file.getRealDimension().x();

	if (maxHeight == 0)
		_maxHeight = _file.getRealDimension().y();
	else
		_maxHeight = maxHeight;

	_sizePlane = _file.getRealDimension().y()*_file.getRealDimension().z();	

	_maxNumPlanes = 3;
	_freeSlots = _maxNumPlanes;

	if (cudaSuccess != cudaHostAlloc((void**)&_memoryPlane, _sizePlane*_maxNumPlanes*sizeof(float), cudaHostAllocDefault))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory"<<std::endl;
		throw;
	}

	_cachePlanes = new cache_plane_t[_maxNumPlanes];

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
	delete[] _cachePlanes;	
}

float * ControlPlaneCache::getAndBlockPlane(int plane)
{
	#ifndef NDEBUG
	if (plane > _maxPlane)
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
		else if (it->second->refs == -1)
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
	vmml::vector<3, int> dim = _file.getRealDimension(); 

	_file.readPlane(data, vmml::vector<3, int>(plane,0,0), vmml::vector<3, int>(plane, _maxHeight, dim.z()));

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
				c->refs = -1;

				readPlane(c->data, c->id);
				_currentPlanes.insert(std::make_pair<int, cache_plane_t *>(c->id, c));

				std::sort(_lruPlanes.begin(), _lruPlanes.end(), compareCachePlane);
			}

		_fullSlots.unlock();
	}
	
}

}

