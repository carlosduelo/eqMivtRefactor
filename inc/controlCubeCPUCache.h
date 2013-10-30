/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CONTROL_CUBE_CPU_CACHE_H
#define EQ_MIVT_CONTROL_CUBE_CPU_CACHE_H

#include <controlPlaneCache.h>

namespace eqMivt
{
class  ControlCubeCPUCache : public ControlElementCache<index_node_t> 
{
	private:
		double	_memoryAviable;
		int		_maxNumCubes;
		float	_memoryOccupancy;

		vmml::vector<3, int> _offset;
		vmml::vector<3, int> _nextOffset;
		int		_dimCube;
		int		_nLevels;
		int		_levelCube;
		int		_nextnLevels;
		int		_nextLevelCube;

		ControlPlaneCache _planeCache;

		virtual bool _readElement(NodeLinkedList<index_node_t> * element);

		virtual bool _threadInit();

		virtual void _threadStop();

		virtual void _freeCache();

		virtual void _reSizeCache();
	public:

		virtual ~ControlCubeCPUCache() {};

		/* Read planes from [min,max) */
		bool initParameter(std::vector<std::string> file_parameters, float memoryOccupancy);

		bool freeCacheAndPause();

		bool reSizeCacheAndContinue(vmml::vector<3,int> offset, vmml::vector<3,int> max, int levelCube, int nLevels);

		// NO SAFE CALLS
		int		getCubeLevel() { return _levelCube; }
		int		getnLevels() { return _nLevels; }
		int		getDimCube() { return _dimCube; }
		vmml::vector<3,int> getRealCubeDim(){ return vmml::vector<3,int>(_dimCube, _dimCube, _dimCube); }

		vmml::vector<2,int>	getPlaneDim() { return _planeCache.getPlaneDim();}
		vmml::vector<3,int> getMinCoord() { return _planeCache.getMinCoord(); }
		vmml::vector<3,int> getMaxCoord() { return _planeCache.getMaxCoord(); }

		int	getMaxPlane() { return _planeCache.getMaxPlane(); }
		int	getMinPlane() { return _planeCache.getMinPlane(); }
};

}

#endif /*   */
