/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CACHE_H
#define EQ_MIVT_CACHE_H

#include <controlCubeCache.h>

#include <boost/unordered_map.hpp>
#include <lunchbox/lock.h>

namespace eqMivt
{

class Cache
{
	struct cube_cached
	{
		int refs;
		float * cube;
	};

	private:
		lunchbox::Lock	_lock;
		ControlCubeCache * _cubeCache;
		boost::unordered_map<index_node_t, cube_cached> _cubes;

		int	_rayCastingLevel;
		float **	_tableCubes;
		float **	_tableCubesGPU;
		int			_sizeTable;
		index_node_t _minValue;

	public:
		Cache();

		bool init(ControlCubeCache * cubeCache);

		void destroy();

		void setRayCastingLevel(int rayCastingLevel) { _rayCastingLevel = rayCastingLevel; }

		void startFrame();

		void finishFrame();

		void pushCubes(visibleCube_t * cube);

		void popCubes(index_node_t id);

		bool isInit() { return _cubeCache != 0; }

		float ** syncAndGetTableCubes(cudaStream_t stream);
};

}
#endif
