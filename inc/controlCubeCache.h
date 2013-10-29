/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CONTROL_CUBE_CACHE_H
#define EQ_MIVT_CONTROL_CUBE_CACHE_H

#include <controlPlaneCache.h>

#include <linkedList.h>

#include <cuda_runtime.h>

#include <queue>

namespace eqMivt
{

struct pending_cube_t
{
	index_node_t			id;
	vmml::vector<3, int>	coord;
};

struct find_pending_cube
{
	index_node_t id;
	find_pending_cube(index_node_t id) : id(id) {}
	bool operator () ( const pending_cube_t& m ) const
	{
		return m.id == id;
	}
};

class  ControlCubeCache : public ControlCache 
{
	private:
		LinkedList<index_node_t>			_lruCubes;

		boost::unordered_map<index_node_t, NodeCube_t *>	_currentCubes;
		std::vector< pending_cube_t >						_pendingCubes;

		std::queue<index_node_t>							_readingCubes;

		cudaStream_t	_stream;
		
		int		_freeSlots;
		int		_maxNumCubes;
		int		_maxCubes;
		float *	_memoryCubes;

		vmml::vector<3, int> _offset;
		vmml::vector<3, int> _nextOffset;
		int		_sizeCubes;
		int		_dimCube;
		int		_nLevels;
		int		_levelCube;
		int		_nextnLevels;
		int		_nextLevelCube;

		#ifdef TIMING
		double _searchCubeN;
		double _insertCubeN;
		double _readingCubeN;
		double _readingCube;
		double _searchCube;
		double _insertCube;
		#endif


		device_t _device;

		ControlPlaneCache * _planeCache;

		lunchbox::Condition	_emptyPendingCubes;
		lunchbox::Condition	_fullSlots;

		bool readCube(NodeCube_t * c);

		void _addNewCube(index_node_t cube);

		virtual void _threadWork();

		virtual bool _threadInit();

		virtual void _threadStop();

		virtual void _freeCache();

		virtual void _reSizeCache();

	public:
		virtual ~ControlCubeCache() {};

		bool initParameter(ControlPlaneCache * planeCache, device_t device);

		bool freeCacheAndPause();

		bool reSizeCacheAndContinue(int nLevels, int levelCube, vmml::vector<3, int> offset);

		float * getAndBlockCube(index_node_t cube);

		void	unlockCube(index_node_t cube);

		// NO SAFE CALLS
		int		getCubeLevel() { return _levelCube; }

		int		getDimCube() { return _dimCube; }
};

}

#endif /*EQ_MIVT_CONTROL_CUBE_CACHE_H*/

