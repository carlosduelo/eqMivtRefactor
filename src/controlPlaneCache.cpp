/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlPlaneCache.h>

#include <memoryCheck.h>

#include <cuda_runtime.h>

#define READING -1
#define WAITING 10

namespace eqMivt
{

bool ControlPlaneCache::initParameter(std::vector<std::string> file_parameters)
{
	bool result = _file.init(file_parameters);

	_min = vmml::vector<3,int>(0,0,0);
	_max = vmml::vector<3,int>(0,0,0);
	_minFuture = vmml::vector<3,int>(0,0,0);
	_maxFuture = vmml::vector<3,int>(0,0,0);

	_memoryAviable = 0;
	_memoryPlane = 0;
	_currentPlanes.clear();
	_pendingPlanes.clear();

	_freeSlots = 0;
	_maxNumPlanes = 0;
	_sizePlane = 0;

	_end = false;
	_resize = false;
	_lastPlane = 0;

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
			_memoryAviable *=0.7;
		}
		if (cudaSuccess != cudaHostAlloc((void**)&_memoryPlane, _memoryAviable, cudaHostAllocDefault))
		{                                                                                               
			std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}

	}

	_maxNumPlanes = _memoryAviable / (_sizePlane * sizeof(float));
	while((_maxNumPlanes*_sizePlane*sizeof(float) + _maxNumPlanes*sizeof(NodeLinkedList)) > _memoryAviable)
	{
		_maxNumPlanes -= 10;
	}

	_freeSlots = _maxNumPlanes;

	_lruPlanes.reSize(_maxNumPlanes);
	_currentPlanes.clear();
	_pendingPlanes.clear();

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

void ControlPlaneCache::exit()
{
	if (_memoryPlane != 0)
		if(cudaSuccess != cudaFreeHost((void*)_memoryPlane))
		{
			std::cerr<<"Control Plane Cache, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}
	_file.close();

	_lockEnd.unset();


	lunchbox::Thread::exit();
}

ControlPlaneCache::~ControlPlaneCache()
{
}

float * ControlPlaneCache::getAndBlockPlane(int plane)
{
	_lockEnd.set();
		if (_end)
		{
			_lockEnd.unset();
			return 0;
		}
	_lockEnd.unset();

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
	_lockEnd.set();
		if (_end)
		{
			_lockEnd.unset();
			return;
		}
	_lockEnd.unset();
	_lockResize.lock();
		if (_resize)
		{
			_lockResize.wait();
			_lockResize.unlock();
			return;
		}
	_lockResize.unlock();

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
		bool prefetching = false;
		int plane = -1;
		_emptyPendingPlanes.lock();
			if (_pendingPlanes.size() == 0)
				if (!_emptyPendingPlanes.timedWait(WAITING))
				{
					prefetching = true;
					plane = _lastPlane + 1 < _max.x() ? _lastPlane + 1 : _min.x();
				}

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
				_emptyPendingPlanes.unlock();
				exit();
			}
			_lockEnd.unset();

			if (!prefetching)
			{
				plane = _pendingPlanes.front();
				_pendingPlanes.erase(_pendingPlanes.begin());
			}
		_emptyPendingPlanes.unlock();

		_fullSlots.lock();
			if (_freeSlots == 0)
			{
				if (!prefetching)
					_fullSlots.wait();
				else
					if (!_fullSlots.timedWait(WAITING))
					{
						continue;
					}
			}

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
				_fullSlots.unlock();
				exit();
			}
			_lockEnd.unset();

			NodeLinkedList * c = 0;
			c = _lruPlanes.getFirstFreePosition();

			boost::unordered_map<int, NodeLinkedList *>::iterator it;
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

			it = _currentPlanes.find(plane);
			if (it == _currentPlanes.end())
			{
				_freeSlots--;
				
				c->id = plane;
				c->refs = READING;

				readPlane(_memoryPlane + (c->element * _sizePlane), c->id);
				_currentPlanes.insert(std::make_pair<int, NodeLinkedList *>(c->id, c));
				_lruPlanes.moveToLastPosition(c);
				_lastPlane = c->id;
			}

		_fullSlots.unlock();
	}
	
}

}
