/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_OCTREE_CONTAINER_H
#define EQ_MIVT_OCTREE_CONTAINER_H

#include <typedef.h>

#include <vmmlib/vector.hpp>

#include <fstream>
#include <iostream>


namespace eqMivt
{

struct octreePosition_t
{
	vmml::vector<3, int>	start;
	vmml::vector<3, int>	end;
	std::vector<float>		isos;
	std::vector<int>		index;
	std::vector<int>		maxHeight;
	int						cubeLevel;
	int						rayCastingLevel;
	int						nLevels;
	int						maxLevel;

	friend std::ostream& operator<<(std::ostream& os, const octreePosition_t& o)
	{
		os<<"Octree:"<<std::endl;
		os<<o.start<<" "<<o.end<<std::endl;
		os<<"Isosurfaces ";
		for(std::vector<float>::const_iterator it=o.isos.begin(); it!=o.isos.end(); it++)
			os<<*it<<" ";
		os<<std::endl;
		os<<"nLevels "<<o.nLevels<<std::endl;
		os<<"maxLevel "<<o.maxLevel<<std::endl;
		os<<"Cube Level "<<o.cubeLevel<<std::endl;
		os<<"Ray casting level "<<o.rayCastingLevel<<std::endl;
		return os;
	}
};

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
		int getIsosurface();

		index_node_t * getOctree() { return _octree; }
		int	* getSizes(){ return _sizes; }
		
};

}

#endif /* EQ_MIVT_OCTREE_CONTAINER_H */
