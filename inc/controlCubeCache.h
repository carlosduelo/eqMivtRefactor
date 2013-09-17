/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CONTROL_CUBE_CACHE_H
#define EQ_MIVT_CONTROL_CUBE_CACHE_H

#include <controlPlaneCache.h>

#include <typedef.h>

namespace eqMivt
{

struct cache_cube_t
{
	index_node_t id;
	float * data;
	int		refs;
	std::time_t timestamp;
};

class  ControlCubeCache : public lunchbox::Thread
{
	private:
		cache_cube_t	*				_cacheCubes;
		std::vector<cache_cube_t *>		_lruCubes;

		boost::unordered_map<index_node_t, cache_cube_t *>	_currentCubes;
		std::vector<index_node_t>							_pendingCubes;
		
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

		lunchbox::Lock		_currentCubessLock;
		lunchbox::Condition	_emptyPendingCubes;
		lunchbox::Condition	_fullSlots;

		void reSizeStructures();
	public:
		virtual ~ControlCubeCache();

		bool initParameter(ControlPlaneCache * planeCache);

		virtual void run();

		bool reSize(int nLevels, int levelCube, vmml::vector<3,int> offset );

		void stopProcessing();

		float * getAndBlockCube(index_node_t cube);

		void	unlockCube(index_node_t cube);
};

}

#endif /*EQ_MIVT_CONTROL_CUBE_CACHE_H*/

