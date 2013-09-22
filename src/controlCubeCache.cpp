/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlCubeCache.h>

#include <mortonCodeUtil_CPU.h>

#define PROCESSING -1
#define PROCESSED -2

namespace eqMivt
{
bool ComparePendingCubes(pending_cube_t a, pending_cube_t b)
{
	if (a.coord.x() == b.coord.x())
		return a.coord.z() < b.coord.z();
	else 
		return a.coord.x() < b.coord.x();
}

bool ControlCubeCache::initParameter(ControlPlaneCache * planeCache)
{
	_planeCache = planeCache;

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

	if (cudaSuccess != cudaStreamCreate(&_stream))
	{
		std::cerr<<"Control Cube Cache, error creating cuda stream: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
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

void ControlCubeCache::exit()
{
	if (_memoryCubes != 0)
		if(cudaSuccess != cudaFree((void*)_memoryCubes))
		{
			std::cerr<<"Control Cubes Cache, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}
	if (cudaSuccess != cudaStreamDestroy(_stream))
	{
		std::cerr<<"Control Cube Cache, error destroying cuda steam: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}

	_lockEnd.unset();

	lunchbox::Thread::exit();
}

ControlCubeCache::~ControlCubeCache()
{
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

	if (_memoryCubes != 0)
		if (cudaSuccess != cudaFree((void*)_memoryCubes))
		{                                                                                               
			std::cerr<<"Control Cube Cache, error resizing cache: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}
	size_t total = 0;
	size_t free = 0;

	if (cudaSuccess != cudaMemGetInfo(&free, &total))
	{
		std::cerr<<"Control Cube Cache, error resizing cache: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
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
		std::cerr<<"Control Cube Cache, error resizing cache: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}

	_freeSlots = _maxNumCubes;

	_lruCubes.reSize(_maxNumCubes);
	_currentCubes.clear();
	_pendingCubes.clear();
	std::queue<index_node_t> emptyQ;
	std::swap(_readingCubes, emptyQ);

	_resize = false;
}

float * ControlCubeCache::getAndBlockCube(index_node_t cube)
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

	float * dcube = 0;
	boost::unordered_map<index_node_t, NodeLinkedList * >::iterator it;

	_fullSlots.lock();

	it = _currentCubes.find(cube);
	if (it != _currentCubes.end())
	{

		if (it->second->refs != PROCESSING)
		{
			if (it->second->refs == 0)
				_freeSlots--;
			else if (it->second->refs == PROCESSED)
				it->second->refs = 0;

			it->second->refs += 1;
			dcube = _memoryCubes + it->second->element*_sizeCubes;
		}
	}
	else
	{
		_emptyPendingCubes.lock();
			if (std::find_if(_pendingCubes.begin(), _pendingCubes.end(), find_pending_cube(cube)) == _pendingCubes.end())
			{
				pending_cube_t pcube;
				pcube.id = cube;
				pcube.coord = getMinBoxIndex2(cube, _levelCube, _nLevels); 
				_pendingCubes.push_back(pcube);
				std::sort(_pendingCubes.begin(), _pendingCubes.end(), ComparePendingCubes);
			}
			if (_pendingCubes.size() == 1)
				_emptyPendingCubes.signal();
		_emptyPendingCubes.unlock();
	}

	_fullSlots.unlock();

	return dcube;
}

void ControlCubeCache::unlockCube(index_node_t cube)
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

	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;

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

	return true;
}

bool ControlCubeCache::readCube(NodeLinkedList * c)
{
	vmml::vector<3, int> coordS = getMinBoxIndex2(c->id, _levelCube, _nLevels) + _offset - vmml::vector<3, int>(CUBE_INC, CUBE_INC, CUBE_INC);
	vmml::vector<3, int> coordE = coordS + vmml::vector<3, int>(_dimCube, _dimCube, _dimCube);

	vmml::vector<2, int> planeDim = _planeCache->getPlaneDim();
	vmml::vector<3, int> minC = _planeCache->getMinCoord();
	vmml::vector<3, int> maxC = _planeCache->getMaxCoord();

	std::vector<int>::iterator it = c->pendingPlanes.begin(); 

	while (it != c->pendingPlanes.end())
	{
		float * start = (_memoryCubes + c->element*_sizeCubes) + _dimCube * _dimCube * abs(coordS.x() - (*it));
		float * plane = _planeCache->getAndBlockPlane(*it);

		if (plane != 0)
		{
			start += coordS.z() < 0 ? abs(coordS.z()) : 0;
			plane += coordS.z() <= minC.z() ? 0 : coordS.z() - minC.z();
			int dimC = _dimCube;
			#if 0
			if (coordS.z() < 0)
				dimC -= abs(coordS.z()) + minC.z();
			else
			#endif
			dimC -= coordS.z() < minC.z() ? abs(minC.z() - coordS.z()) : 0;
			dimC -= coordE.z() > maxC.z() ? coordE.z() - maxC.z() : 0;

			int sL = coordS.y() < minC.y() ? abs(minC.y() - coordS.y()) : 0;
			int lL = coordE.y() > maxC.y() ? _dimCube - (coordE.y() - maxC.y()) : _dimCube;

			for(int i = sL; i < lL; i++)
			{
				int planeI = i+(coordS.y() - minC.y());
				if (cudaSuccess != cudaMemcpyAsync((void*)(start + i*_dimCube), (void*)(plane + planeI*planeDim.y()), dimC*sizeof(float), cudaMemcpyHostToDevice, _stream))
				{
					std::cerr<<"Control Cube Cache, error copying cube to GPU: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
					throw;
				}
			}

			if(cudaSuccess != cudaStreamSynchronize(_stream))
			{
				std::cerr<<"Control Cubes Cache, stream synchronization: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
				throw;
			}

			_planeCache->unlockPlane(*it);
			it = c->pendingPlanes.erase(it);
		}

		if (it != c->pendingPlanes.end())
			it++;
	}


	return c->pendingPlanes.empty();
}

void ControlCubeCache::run()
{
	while(1)
	{
		index_node_t cube = 0;
		if (!_readingCubes.empty())
		{
			cube = _readingCubes.front();
			_readingCubes.pop();

			_fullSlots.lock();
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

				boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
				it = _currentCubes.find(cube);
				if (it != _currentCubes.end())
				{
					if (!readCube(it->second))
						_readingCubes.push(it->second->id);
					else
						it->second->refs = PROCESSED;

					_lruCubes.moveToLastPosition(it->second);
				}
				else
				{
					std::cerr<<"Control Cube Cache, error reding cube that is not in current cubes"<<std::endl;
					throw;
				}
			_fullSlots.unlock();
		}
		else
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
					_emptyPendingCubes.unlock();
					exit();
				}
				_lockEnd.unset();

				pending_cube_t pcube = _pendingCubes.front();
				_pendingCubes.erase(_pendingCubes.begin());
				cube = pcube.id;
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
					_fullSlots.unlock();
					exit();
				}
				_lockEnd.unset();

				NodeLinkedList * c = _lruCubes.getFirstFreePosition();

				boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
				it = _currentCubes.find(c->id);
				if (it != _currentCubes.end())
					_currentCubes.erase(it);	

				#ifndef NDEBUG
				if (c->refs != 0 || c->pendingPlanes.size() != 0)
				{
					std::cerr<<"Control Plane Cache, unistable state, free cube slot with references "<<c->id<<" refs "<<c->refs<<std::endl;
					throw;
				}
				#endif

				it = _currentCubes.find(cube);
				if (it == _currentCubes.end())
				{
					_freeSlots--;
					
					c->id = cube;
					c->refs = PROCESSING;
					vmml::vector<3, int> coord = getMinBoxIndex2(cube, _levelCube, _nLevels) + _offset;
					int minPlane = coord.x() - CUBE_INC;
					int maxPlane = minPlane + _dimCube;
					for(int i=minPlane; i<maxPlane; i++)
						if (i>=_planeCache->getMinPlane() && i< _planeCache->getMaxPlane())
							c->pendingPlanes.push_back(i);

					if (cudaSuccess != cudaMemsetAsync((void*)(_memoryCubes + c->element*_sizeCubes), 0, _dimCube*_dimCube*_dimCube*sizeof(float), _stream))
					{
						std::cerr<<"Control Cube Cache, error copying cube to GPU: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
						throw;
					}

					if (!readCube(c))
						_readingCubes.push(c->id);
					else
						c->refs = PROCESSED;

					_currentCubes.insert(std::make_pair<index_node_t, NodeLinkedList *>(c->id, c));
					_lruCubes.moveToLastPosition(c);

				}

			_fullSlots.unlock();
		}
	}
}

}

