/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_OCTREE_H
#define EQ_MIVT_OCTREE_H

#include <typedef.h>
#include <octreeContainer.h>

namespace eqMivt
{

class Octree
{
	private:
		OctreeContainer *	_oc;

		float		*		_xGrid;
		float		*		_yGrid;
		float		*		_zGrid;

		int					_currentOctree;
		index_node_t *		_memoryOctree;
		index_node_t ** 	_octree;
		int	*				_sizes;

		device_t			_device;

	public:
		Octree();

		bool init(OctreeContainer * oc, device_t device);
	
		bool loadCurrentOctree();

		void stop();

		index_node_t ** getOctree(){ return _octree; }
		int * getSizes(){ return _sizes; }
		float * getxGrid();
		float * getyGrid();
		float * getzGrid();

		vmml::vector<3, int> getStartCoord(){return _oc->getStartCoord();}
		vmml::vector<3, int> getEndCoord(){return _oc->getEndCoord();}
		vmml::vector<3, int> getRealDim(){return _oc->getEndCoord() - _oc->getStartCoord();}
		vmml::vector<3, float> getGridStartCoord(){return _oc->getGridStartCoord();}
		vmml::vector<3, float> getGridEndCoord(){return _oc->getGridEndCoord();}
		vmml::vector<3, float> getGridRealDimVolume(){return _oc->getGridRealDimVolume();}
		int getnLevels() {return _oc->getnLevels() ;}
		int getmaxLevel(){return _oc->getmaxLevel() ;}
		int	getCubeLevel(){return _oc->getCubeLevel() ;}
		int getRayCastingLevel(){return _oc->getRayCastingLevel() ;}
		int	getMaxHeight(){return _oc->getMaxHeight() ;}
		float getGridMaxHeight(){return _oc->getGridMaxHeight() ;}
		float getIsosurface(){return _oc->getIsosurface() ;}
};

}
#endif
