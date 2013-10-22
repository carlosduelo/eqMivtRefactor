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
}

void Cache::startFrame()
{
	_cubes.clear();
}

void Cache::finishFrame()
{
	if (_cubes.size() != 0)
	{
		std::cerr<<"Error cache not working"<<std::endl;
		throw;
	}
}

bool Cache::init(ControlCubeCache * cubeCache)
{
	if (cubeCache == 0)
		return false;

	_cubeCache = cubeCache;

	_rayCastingLevel = 0;

	return true;
}

void Cache::pushCubes(VisibleCubes * vc)
{
	#ifndef NDEBUG
	if (_cubeCache->getCubeLevel() > _rayCastingLevel)
	{
		std::cerr<<"Cache, error ray casting cube level has to be >= cache cube level"<<std::endl;
		throw;
	}
	#endif

	std::vector<int> cubes = vc->getListCubes(CUBE);

	for(std::vector<int>::iterator it = cubes.begin(); it != cubes.end(); ++it)
	{
		visibleCube_t * cube = vc->getCube(*it);

		index_node_t idCube = cube->id >> (3*(_rayCastingLevel - _cubeCache->getCubeLevel()));

		boost::unordered_map<index_node_t, cube_cached>::iterator itC = _cubes.find(idCube);

		if (itC != _cubes.end())
		{
			cube->cubeID = idCube;
			cube->state = itC->second.state;
			cube->data = itC->second.cube;
		}
		else
		{
			float * d = _cubeCache->getAndBlockCube(idCube);

			cube->cubeID = idCube;
			cube->state = d == 0 ? CUBE : CACHED;
			cube->data = d;

			cube_cached c;
			c.state = cube->state; 
			c.cube = d;
			
			_cubes.insert(std::make_pair<index_node_t, cube_cached>(idCube, c));
		}

	}

}

void Cache::popCubes()
{
	boost::unordered_map<index_node_t, cube_cached>::iterator it=_cubes.begin();

	while(it!=_cubes.end())
	{
		if (it->second.state == CACHED)
				_cubeCache->unlockCube(it->first);

		it++;
	}

	_cubes.clear();
}

}
