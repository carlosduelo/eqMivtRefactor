/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlCubeCache.h>

#include <cuda_runtime.h>

namespace eqMivt
{
bool compareCacheCube(cache_cube_t * a, cache_cube_t * b)
{
	if (a->refs > 0 && b->refs > 0)
		return a->timestamp < b->timestamp;
	else if (a->refs == -1)
		return false;
	else if (b->refs == -1)
		return true;
	else if (a->refs == 0 && b->refs == 0)
		return a->timestamp < b->timestamp;
	else
		return false;//a->refs == 0;
}

bool ControlCubeCache::initParameter(ControlPlaneCache * planeCache)
{
	_planeCache = planeCache;

	_cacheCubes = 0;
	_freeSlots = 0;
	_maxNumCubes = 0;
	_maxCubes = 0;

	_offset.set(0,0,0);
	_nextOffset.set(0,0,0);
	_nextnLevels = -1;
	_nextLevelCube = -1;
	_nLevels = 0;
	_levelCube = 0;
	_dimCube = 0;
	_sizeCubes = 0;
	_memoryCubes = 0;
	
	_end = false;
	_resize = false;
}

void ControlCubeCache::stopProcessing()
{
	_lockEnd.set();
		_emptyPendingCubes.broadcast();
		_fullSlots.broadcast();
		_end = true;
	_lockEnd.unset();

	join();
}

ControlCubeCache::~ControlCubeCache()
{
	_lockResize.lock();
		if (_resize)
			_lockResize.wait();

	if (_cacheCubes != 0)
		delete[] _cacheCubes;
	if (_memoryCubes != 0)
		if(cudaSuccess != cudaFree((void*)_memoryCubes))
		{
			std::cerr<<"Control Cubes Cache, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}
	_lockResize.unlock();
}

void ControlCubeCache::reSizeStructures()
{
	_nLevels = _nextnLevels;
	_levelCube = _nextLevelCube;
	_offset	= _nextOffset;
	_nextnLevels = 0;;
	_nextLevelCube = 0;

	_dimCube = exp2(_nLevels - _levelCube) + 2 * CUBE_INC;

	_sizeCubes = pow(_dimCube, 3); 

	_lruCubes.clear();
	_currentCubes.clear();
	_pendingCubes.clear();

	if (_memoryCubes != 0)
		if (cudaSuccess != cudaFree((void*)_memoryCubes))
		{                                                                                               
			std::cerr<<"Control Cube Cache, error resizing cache"<<std::endl;
			throw;
		}
	size_t total = 0;
	size_t free = 0;

	if (cudaSuccess != cudaMemGetInfo(&free, &total))
	{
		std::cerr<<"Control Cube Cache, error resizing cache"<<std::endl;
		throw;
	}

	float memorySize = (8.0f*free)/10.0f; // Get 80% of free memory

	_maxNumCubes = memorySize/ (_sizeCubes*sizeof(float));
	if (_maxNumCubes == 0)
	{
		std::cerr<<"Control Cube Cache: Memory aviable is not enough "<<memorySize/1024/1024<<" MB"<<std::endl;
		throw;
	}

	if (cudaSuccess != cudaMalloc((void**)&_memoryCubes, _maxNumCubes*_sizeCubes*sizeof(float)))
	{
		std::cerr<<"Control Cube Cache, error resizing cache"<<std::endl;
		throw;
	}

	_freeSlots = _maxNumCubes;

	if (_cacheCubes != 0)
	{
		delete[] _cacheCubes;
	}
	_cacheCubes = 0;
	_cacheCubes = new cache_cube_t[_maxNumCubes];
	if (_cacheCubes == 0)
	{
		std::cerr<<"Control Cube Cache, error resizing cache"<<std::endl;
		throw;
	}
	for(int i=0; i<_maxNumCubes; i++)
	{
		_cacheCubes[i].id = -1;
		_cacheCubes[i].data = _memoryCubes + i*_sizeCubes;
		_cacheCubes[i].refs = 0;
		_cacheCubes[i].timestamp = std::time(0);
		_lruCubes.push_back(&_cacheCubes[i]);
	}

	_resize = false;
}

float * ControlCubeCache::getAndBlockCube(index_node_t cube)
{
	_lockResize.lock();
		if (_resize)
		{
			_lockResize.wait();
			_lockResize.unlock();
			return 0;
		}
	_lockResize.unlock();

	float * dcube = 0;
	boost::unordered_map<index_node_t, cache_cube_t * >::iterator it;

	_fullSlots.lock();

	it = _currentCubes.find(cube);
	if (it != _currentCubes.end())
	{
		if (it->second->refs == 0)
			_freeSlots--;
		else if (it->second->refs == -1)
			it->second->refs = 0;

		it->second->refs += 1;
		it->second->timestamp = std::time(0); 
		dcube = it->second->data;

		std::sort(_lruCubes.begin(), _lruCubes.end(), compareCacheCube);
	}
	else
	{
		_emptyPendingCubes.lock();
			if (std::find(_pendingCubes.begin(), _pendingCubes.end(), cube) == _pendingCubes.end())
				_pendingCubes.push_back(cube);
			if (_pendingCubes.size() == 1)
				_emptyPendingCubes.signal();
		_emptyPendingCubes.unlock();
	}

	_fullSlots.unlock();

	return dcube;
}

void ControlCubeCache::unlockCube(index_node_t cube)
{
	_lockResize.lock();
		if (_resize)
		{
			_lockResize.wait();
			_lockResize.unlock();
			return;
		}
	_lockResize.unlock();

	boost::unordered_map<index_node_t, cache_cube_t *>::iterator it;

	_fullSlots.lock();

	it = _currentCubes.find(cube);
	if (it != _currentCubes.end())
	{
		it->second->refs -= 1;
		if (it->second->refs == 0)
		{
			_freeSlots++;
			_fullSlots.signal();
		}
		
		std::sort(_lruCubes.begin(), _lruCubes.end(), compareCacheCube);

		#ifndef NDEBUG
		if (it->second->refs < 0)
		{
			std::cerr<<"Control Plane Cache, error unlocking cube"<<std::endl;
			throw;
		}
		#endif
	}
	#ifndef NDEBUG
	else
	{
		std::cerr<<"Control Plane Cache, error unlocking cube that not exists"<<std::endl;
		throw;
	}
	#endif

	_fullSlots.unlock();
}

bool ControlCubeCache::reSize(int nLevels, int levelCube, vmml::vector<3, int> offset)
{
	_lockResize.lock();

		_emptyPendingCubes.signal();
		_fullSlots.signal();

		_nextOffset	= offset;
		_nextnLevels = nLevels;
		_nextLevelCube = levelCube;
		_resize = true;

	_lockResize.wait();

	_lockResize.unlock();

}

void ControlCubeCache::run()
{
	while(1)
	{
		_emptyPendingCubes.lock();
			if (_pendingCubes.size() == 0)
				_emptyPendingCubes.wait();

			_lockResize.lock();
				if (_resize)
				{
					reSizeStructures();	
					_lockResize.broadcast();
					_lockResize.unlock();
					_emptyPendingCubes.unlock();
					continue;
				}
			_lockResize.unlock();
				
			_lockEnd.set();
			if (_end)
			{
				_lockEnd.unset();
				_emptyPendingCubes.unlock();
				exit();
			}
			_lockEnd.unset();

			index_node_t  cube = _pendingCubes.front();
			_pendingCubes.erase(_pendingCubes.begin());
		_emptyPendingCubes.unlock();

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

			cache_cube_t * c = _lruCubes.front();

			boost::unordered_map<index_node_t, cache_cube_t *>::iterator it;
			it = _currentCubes.find(c->id);
			if (it != _currentCubes.end())
				_currentCubes.erase(it);	

			#ifndef NDEBUG
			if (c->refs != 0)
			{
				std::cerr<<"Control Plane Cache, unistable state, free cube slot with references "<<c->id<<" refs "<<c->refs<<std::endl;
				while(_lruCubes.size() != 0)
				{
					cache_cube_t * p = _lruCubes[0];
					_lruCubes.erase(_lruCubes.begin());
					std::cerr<<"Plane "<<p->id<<" "<<p->refs<<std::endl;
				}
				throw;
			}
			#endif

			it = _currentCubes.find(cube);
			if (it == _currentCubes.end())
			{
				_freeSlots--;
				
				c->id = cube;
				c->refs = -1;

				//readPlane(c->data, c->id);
				_currentCubes.insert(std::make_pair<index_node_t, cache_cube_t *>(c->id, c));

				std::sort(_lruCubes.begin(), _lruCubes.end(), compareCacheCube);
			}

		_fullSlots.unlock();
	}
}

}
