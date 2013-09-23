/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CACHE_H
#define EQ_MIVT_CACHE_H

#include <controlCubeCache.h>
#include <visibleCubes.h>

namespace eqMivt
{

class Cache
{
	private:
		ControlCubeCache * _cubeCache;

		int	_rayCastingLevel;

		std::vector<index_node_t> _updateCube;

	public:
		bool init(ControlCubeCache * cubeCache);

		void setRayCastingLevel(int rayCastingLevel) { _rayCastingLevel = rayCastingLevel; }

		void pushCubes(VisibleCubes * vc);

		void popCubes();

};

}
#endif
