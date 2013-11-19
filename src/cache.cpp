/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <cache.h>

namespace eqMivt
{

Cache::Cache()
{
	_cubeCache = 0;
	_tableCubes = 0;
	_tableCubesGPU = 0;
	_sizeTable = 0;
	_minValue = 0;
}

void Cache::startFrame()
{
	_lock.set();
	_cubes.clear();
	if (_sizeTable != _cubeCache->getNumCubes())
	{
		_minValue = _cubeCache->getMinValue();
		_sizeTable = _cubeCache->getNumCubes();
		if (_tableCubes != 0)
			delete[] _tableCubes;

		if (cudaSuccess != cudaFree(_tableCubesGPU))
		{
			std::cerr<<"Cache, error table cubes gpu free"<<std::endl;
			throw;
		}

		_tableCubes = new float*[_sizeTable];

		if (cudaSuccess != cudaMalloc((void**)&_tableCubesGPU, _sizeTable*sizeof(float*)))
		{
			std::cerr<<"Cache, error table cubes gpu free"<<std::endl;
			throw;
		}
	}
	memset((void*)_tableCubes, 0, _sizeTable*sizeof(float*)); 
	_lock.unset();
}

void Cache::finishFrame()
{
	_lock.set();

	if (_cubes.size() != 0)
	{
		boost::unordered_map<index_node_t, cube_cached>::iterator it= _cubes.begin();
		while(it != _cubes.end())
		{
			_cubeCache->unlockElement(it->first);
			it++;
		}
		_cubes.clear();
	}

	_lock.unset();
}

bool Cache::init(ControlCubeCache * cubeCache)
{
	if (cubeCache == 0)
		return false;

	_cubeCache = cubeCache;

	_rayCastingLevel = 0;

	return true;
}

void Cache::destroy()
{
	if (_tableCubes != 0)
	{
		delete[] _tableCubes;
		if (cudaSuccess != cudaFree(_tableCubesGPU))
		{
			std::cerr<<"Cache, error table cubes gpu free"<<std::endl;
			throw;
		}
	}
}

void Cache::pushCubes(visibleCube_t * cube)
{
	#ifndef NDEBUG
	if (_cubeCache->getCubeLevel() > _rayCastingLevel)
	{
		std::cerr<<"Cache, error ray casting cube level has to be >= cache cube level"<<std::endl;
		throw;
	}
	#endif

	if (cube->state == CUBE)
	{
		index_node_t idCube = cube->id >> (3*(_rayCastingLevel - _cubeCache->getCubeLevel()));

		_lock.set();
		boost::unordered_map<index_node_t, cube_cached>::iterator itC = _cubes.find(idCube);

		if (itC != _cubes.end())
		{
			cube->idCube = idCube;
			cube->state = CACHED;
			itC->second.refs++;
			_tableCubes[idCube - _minValue] = itC->second.cube;
		}
		else
		{
			float * d = _cubeCache->getAndBlockElement(idCube);

			if (d != 0)
			{
				cube->idCube = idCube;
				cube->state = CACHED;
				_tableCubes[idCube - _minValue] = d;

				cube_cached c;
				c.cube = d;
				c.refs = 1;
				
				_cubes.insert(std::make_pair<index_node_t, cube_cached>(idCube, c));
			}
			else
			{
				cube->idCube = 0;
			}
		}
		
		_lock.unset();
	}
	else
	{
		cube->idCube = 0;
	}
}

void Cache::popCubes(index_node_t id)
{
	_lock.set();
	boost::unordered_map<index_node_t, cube_cached>::iterator it= _cubes.find(id);

	if (it != _cubes.end())
	{
		it->second.refs--;
		if (it->second.refs == 0)
		{
			_cubeCache->unlockElement(it->first);
			_tableCubes[it->first - _minValue] = 0;
			_cubes.erase(it);
		}
	}
	else
	{
		std::cerr<<"Cube not pushed "<<it->first<<std::endl;
	}
	_lock.unset();
}

float ** Cache::syncAndGetTableCubes(cudaStream_t stream)
{
	if (cudaSuccess != cudaMemcpyAsync((void*)_tableCubesGPU, _tableCubes, _sizeTable*sizeof(float*), cudaMemcpyHostToDevice, stream))
	{
		std::cerr<<"Cache, error sync table cubes"<<std::endl;
		throw;
	}
	return _tableCubesGPU;
}
}
