/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CONTROL_CUBE_CACHE_H
#define EQ_MIVT_CONTROL_CUBE_CACHE_H

#include <controlPlaneCache.h>

#include <cuda_runtime.h>

#include <queue>

namespace eqMivt
{

struct cache_cube_t
{
	index_node_t id;
	float * data;
	int		refs;
	std::time_t timestamp;
	std::vector<int> pendingPlanes;
};

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

class  ControlCubeCache : public lunchbox::Thread
{
	private:
		cache_cube_t	*				_cacheCubes;
		std::vector<cache_cube_t *>		_lruCubes;

		boost::unordered_map<index_node_t, cache_cube_t *>	_currentCubes;
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

		lunchbox::Lock		_lockEnd;
		bool				_end;
		lunchbox::Condition	_lockResize;
		bool				_resize;

		ControlPlaneCache * _planeCache;

		lunchbox::Condition	_emptyPendingCubes;
		lunchbox::Condition	_fullSlots;

		void reSizeStructures();

		bool readCube(cache_cube_t * c);

	public:
		virtual ~ControlCubeCache();

		bool initParameter(ControlPlaneCache * planeCache);

		virtual void run();

		bool reSize(int nLevels, int levelCube, vmml::vector<3,int> offset );

		void stopProcessing();

		float * getAndBlockCube(index_node_t cube);

		void	unlockCube(index_node_t cube);

		int		getCubeLevel() { return _levelCube; }
};

}

#endif /*EQ_MIVT_CONTROL_CUBE_CACHE_H*/

