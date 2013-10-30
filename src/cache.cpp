/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <cache.h>

#ifdef TIMING
#include <lunchbox/clock.h>
#endif

namespace eqMivt
{

Cache::Cache()
{
	_cubeCache = 0;
}

void Cache::startFrame()
{
	#ifdef TIMING
	_cOPushN = 0.0;
	_searchOPushN = 0.0;
	_getCubePushN = 0.0;
	_PopN = 0.0;
	_cOPush = 0.0;
	_searchOPush = 0.0;
	_getCubePush = 0.0;
	_Pop = 0.0;
	#endif

	_cubes.clear();
}

void Cache::finishFrame()
{
	if (_cubes.size() != 0)
	{
		std::cerr<<"Error cache not working"<<std::endl;
		throw;
	}
	#ifdef TIMING
	std::cout<<"===== Cache Statistics ====="<<std::endl;
	std::cout<<"Complete time push operation "<<_cOPush<<" seconds"<<std::endl;
	std::cout<<"Complete time search operation "<<_searchOPush<<" seconds"<<std::endl;
	std::cout<<"Complete time geting operation "<<_getCubePush<<" seconds"<<std::endl;
	std::cout<<"Complete time pop operation "<<_Pop<<" seconds"<<std::endl;
	std::cout<<"Average time push operation "<<_cOPush/_cOPushN<<" seconds, operations "<<_cOPushN<<std::endl;
	std::cout<<"Average time search operation "<<_searchOPush/_searchOPushN<<" seconds, operations "<<_searchOPushN<<std::endl;
	std::cout<<"Average time geting operation "<<_getCubePush/_getCubePushN<<" seconds, operations "<<_getCubePushN<<std::endl;
	std::cout<<"Average time pop operation "<<_Pop/_PopN<<" seconds, operations "<<_PopN<<std::endl;
	std::cout<<"=============================="<<std::endl;
	#endif
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
	#ifdef TIMING
	lunchbox::Clock clockC;
	lunchbox::Clock clockO;
	#endif

	for(std::vector<int>::iterator it = cubes.begin(); it != cubes.end(); ++it)
	{
	#ifdef TIMING
	clockC.reset();
	#endif
		visibleCube_t * cube = vc->getCube(*it);

		index_node_t idCube = cube->id >> (3*(_rayCastingLevel - _cubeCache->getCubeLevel()));

	#ifdef TIMING
	clockO.reset();
	#endif
		boost::unordered_map<index_node_t, cube_cached>::iterator itC = _cubes.find(idCube);
	#ifdef TIMING
	_searchOPush += clockO.getTimed()/1000.0;
	_searchOPushN += 1.0;
	#endif

		if (itC != _cubes.end())
		{
			cube->cubeID = idCube;
			cube->state = itC->second.state;
			cube->data = itC->second.cube;
		}
		else
		{
	#ifdef TIMING
	clockO.reset();
	#endif
			float * d = _cubeCache->getAndBlockElement(idCube);
	#ifdef TIMING
	_getCubePush += clockO.getTimed()/1000.0;
	_getCubePushN += 1.0;
	#endif

			cube->cubeID = idCube;
			cube->state = d == 0 ? CUBE : CACHED;
			cube->data = d;

			cube_cached c;
			c.state = cube->state; 
			c.cube = d;
			
			_cubes.insert(std::make_pair<index_node_t, cube_cached>(idCube, c));
		}
	#ifdef TIMING
	_cOPush += clockC.getTimed()/1000.0;
	_cOPushN+=1.0;
	#endif
	}

}

void Cache::popCubes()
{
	boost::unordered_map<index_node_t, cube_cached>::iterator it=_cubes.begin();
	#ifdef TIMING
	lunchbox::Clock clock;
	#endif

	while(it!=_cubes.end())
	{
		if (it->second.state == CACHED)
		{
			#ifdef TIMING
			_PopN += 1.0;
			clock.reset();
			#endif
			_cubeCache->unlockElement(it->first);
			#ifdef TIMING
			_Pop += clock.getTimed()/1000.0; 
			#endif
		}

		it++;
	}

	_cubes.clear();
}

}
