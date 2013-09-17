/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlPlaneCache.h>

#include <cuda_runtime.h>

namespace eqMivt
{

bool ControlPlaneCache::initParamenter(std::vector<std::string> file_parameters, int maxHeight)
{
	_file.init(file_parameters);
	
	_maxPlane = 1000;

	_sizePlane = 100;	
	_maxNumPlanes = 3;
	_freeSlots = _maxNumPlanes;

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
	std::cout<<"STOP PROCESSING"<<std::endl;
	_lockEnd.set();
		_emptyPendingPlanes.broadcast();
		_fullSlots.broadcast();
		_end = true;
	_lockEnd.unset();

	std::cout<<"WAITING"<<std::endl;
	join();
	std::cout<<"FINISH"<<std::endl;
}

ControlPlaneCache::~ControlPlaneCache()
{
	std::cout<<"Exit processing planes"<<std::endl;

	if (_memoryPlane != 0)
		if(cudaSuccess != cudaFreeHost((void*)_memoryPlane))
		{
			std::cerr<<"Control Plane Cache, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}
}

#if 0
void ControlPlaneCache::addPlane(int plane)
{
	#ifndef NDEBUG
	if (plane > _maxPlane)
	{
		std::cerr<<"Control Plane Cache, error adding plane that not exists"<<std::endl;
		return;
	}
	#endif

	boost::unordered_map<int, cache_plane_t>::iterator it;


	std::cout<<"ADDING PLANE "<<plane<<std::endl;

	_fullSlots.lock();

	it = _currentPlanes.find(plane);
	if (it == _currentPlanes.end())
	else
	{
		it->second.timestamp = std::time(0); 
	}

	_fullSlots.unlock();
}

void ControlPlaneCache::addPlanes(std::vector<int> planes)
{
	for(std::vector<int>::iterator it = planes.begin(); it!=planes.end(); ++it)
		addPlane(*it);
}
#endif

float * ControlPlaneCache::getAndBlockPlane(int plane)
{
	float * dplane = 0;
	boost::unordered_map<int, cache_plane_t>::iterator it;

	_fullSlots.lock();

	it = _currentPlanes.find(plane);
	if (it != _currentPlanes.end())
	{
		//std::cout<<"GET AND BLOCK "<<plane<<std::endl;
		if (it->second.refs == 0)
			_freeSlots--;
		else if (it->second.refs == -1)
			it->second.refs = 0;

		it->second.refs += 1;
		it->second.timestamp = std::time(0); 
		dplane = it->second.data;
	}
	else
	{
		//std::cout<<"PENDING "<<plane<<std::endl;
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
	boost::unordered_map<int, cache_plane_t>::iterator it;

	_fullSlots.lock();

	//std::cout<<"UNLOCK "<<plane<<std::endl;
	it = _currentPlanes.find(plane);
	if (it != _currentPlanes.end())
	{
		it->second.refs -= 1;
		if (it->second.refs == 0)
		{
			_freeSlots++;
			_fullSlots.signal();
		}

		#ifndef NDEBUG
		if (it->second.refs < 0)
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
			//std::cout<<"Processing plane "<<plane<<std::endl;
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

			cache_plane_t c = _lruPlanes.top();
			_lruPlanes.pop();

			boost::unordered_map<int, cache_plane_t>::iterator it;
			it = _currentPlanes.find(c.id);
			if (it != _currentPlanes.end())
				_currentPlanes.erase(it);	

			#ifndef NDEBUG
			it = _currentPlanes.find(plane);
			if (it != _currentPlanes.end())
			{
				std::cerr<<"Control Plane Cache, error processing a plane which is already processed"<<std::endl;
			}
			#endif

			#ifndef NDEBUG
			if (c.refs != 0)
			{
				std::cerr<<"Control Plane Cache, unistable state, free plane slot with references"<<std::endl;
				throw;
			}
			#endif

			_freeSlots--;
			
			c.id = plane;
			c.refs = -1;

			readPlane(c.data, c.id);
			_lruPlanes.push(c);
			_currentPlanes.insert(std::make_pair<int, cache_plane_t>(c.id, c));

			//std::cout<<"Plane processed "<<plane<<" "<<c.data<<std::endl;
		_fullSlots.unlock();
	}
	
}

}

