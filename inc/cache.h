/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CACHE_H
#define EQ_MIVT_CACHE_H

#include <controlCubeCache.h>
#include <visibleCubes.h>

#include <boost/unordered_map.hpp>

namespace eqMivt
{

class Cache
{
	struct cube_cached
	{
		statusCube state;
		float * cube;
	};

	private:
		ControlCubeCache * _cubeCache;
		boost::unordered_map<index_node_t, cube_cached> _cubes;

		int	_rayCastingLevel;

	public:
		Cache();

		bool init(ControlCubeCache * cubeCache);

		void setRayCastingLevel(int rayCastingLevel) { _rayCastingLevel = rayCastingLevel; }

		void startFrame();

		void finishFrame();

		void pushCubes(VisibleCubes * vc);

		void popCubes();

		bool isInit() { return _cubeCache != 0; }
};

}
#endif
