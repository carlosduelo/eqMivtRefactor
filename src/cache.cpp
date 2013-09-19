/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <cache.h>

namespace eqMivt
{

bool Cache::init(ControlCubeCache * cubeCache)
{
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
	if (_updateCube.size() != 0)
	{
		std::cerr<<"Cache, please pop before push again"<<std::endl;
		throw;
	}
	#endif

	std::vector<int> cubes = vc->getListCubes(CUBE);

	for(std::vector<int>::iterator it = cubes.begin(); it != cubes.end(); ++it)
	{
		visibleCube_t cube = vc->getCube(*it);

		index_node_t idCube = cube.id >> (3*(_rayCastingLevel - _cubeCache->getCubeLevel()));

		float * d = _cubeCache->getAndBlockCube(idCube);

		if (d != 0)
		{
			cube.cubeID = idCube;
			cube.state = CACHED;
			cube.data = d;
			_updateCube.push_back(cube);
		}
	}

	vc->updateVisibleCubes(_updateCube);

}

void Cache::popCubes()
{
	for(std::vector<visibleCube_t>::iterator it=_updateCube.begin(); it!=_updateCube.end(); ++it)
	{
		_cubeCache->unlockCube(it->cubeID);
	}

	_updateCube.clear();
}

}
