/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CONTROL_CUBE_CACHE_H
#define EQ_MIVT_CONTROL_CUBE_CACHE_H

#include <controlCubeCPUCache.h>

#include <cuda_runtime.h>

namespace eqMivt
{

class  ControlCubeCache : public ControlElementCache<index_node_t>
{
	private:
		int		_maxNumCubes;

		vmml::vector<3, int> _offset;
		vmml::vector<3, int> _nextOffset;
		int		_sizeCubes;
		int		_dimCube;
		int		_nLevels;
		int		_levelCube;
		int		_nextnLevels;
		int		_nextLevelCube;

		vmml::vector<3, int> _maxC;

		device_t _device;
		cudaStream_t _stream;

		ControlCubeCPUCache * _cpuCache;

		virtual bool _readElement(NodeLinkedList<index_node_t> * element);

		virtual bool _threadInit();

		virtual void _threadStop();

		virtual void _freeCache();

		virtual void _reSizeCache();
	public:
		virtual ~ControlCubeCache() {};

		bool initParameter(ControlCubeCPUCache * cpuCache, device_t device);

		bool freeCacheAndPause();

		bool reSizeCacheAndContinue(int nLevels, int levelCube, vmml::vector<3, int> offset);

		bool stopCache();

		bool checkCubeInside(index_node_t id);

		// NO SAFE CALLS
		int		getCubeLevel() { return _levelCube; }

		int		getDimCube() { return _dimCube; }
};

}

#endif /*EQ_MIVT_CONTROL_CUBE_CACHE_H*/

