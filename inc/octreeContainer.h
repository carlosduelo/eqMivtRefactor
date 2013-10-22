/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_OCTREE_CONTAINER_H
#define EQ_MIVT_OCTREE_CONTAINER_H

#include <typedef.h>

#include <fstream>
#include <iostream>


namespace eqMivt
{


class OctreeContainer
{
	private:
		std::ifstream _file;

		int		_numOctrees;
		int		_currentPosition;
		int		_currentIsosurface;
		std::vector<octreePosition_t> _octrees;
		vmml::vector<3, int> _realDimVolume;

		int *	_desp;

		float * _xGrid;
		float * _yGrid;
		float * _zGrid;

		index_node_t * _octree;
		int	*			_sizes;
	
		void _setBestCubeLevel();
		void _readCurrentOctree();
	public:

		OctreeContainer();

		bool init(std::string file_name);

		void stop();

		bool checkLoadNextPosition();
		bool checkLoadPreviusPosition();

		bool checkLoadNextIsosurface();
		bool checkLoadPreviusIsosurface();

		bool loadNextPosition();
		bool loadPreviusPosition();

		bool loadNextIsosurface();
		bool loadPreviusIsosurface();

		float * getXGrid(){ return _xGrid; }
		float * getYGrid(){ return _yGrid; }
		float * getZGrid(){ return _zGrid; }

		vmml::vector<3, int> getRealDimVolume(){ return _realDimVolume; }

		int	getCurrentOctree();

		int getNumOctrees(){ return _numOctrees; }
		vmml::vector<3, int> getStartCoord();
		vmml::vector<3, int> getEndCoord();
		int getnLevels();
		int getmaxLevel();
		int	getCubeLevel();
		int getRayCastingLevel();
		int	getMaxHeight();
		float getIsosurface();

		index_node_t * getOctree() { return _octree; }
		int	* getSizes(){ return _sizes; }
		
};

}

#endif /* EQ_MIVT_OCTREE_CONTAINER_H */
