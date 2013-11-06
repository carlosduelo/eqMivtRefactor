/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_OCTREE_MANAGER_H
#define EQ_MIVT_OCTREE_MANAGER_H

#include <octree.h>

#include <boost/unordered_map.hpp>

namespace eqMivt
{

class OctreeManager
{
	private:
		OctreeContainer _oc;
		boost::unordered_map<device_t, Octree *> _octrees;
	
	public:
		bool init(std::string file_name);

		void stop();

		int getNumOctrees(){ return _oc.getNumOctrees(); }
		int getCurrentOctree();

		bool existsDevice(device_t device);

		Octree * getOctree(device_t device);

		bool checkLoadNextPosition() {return _oc.checkLoadNextPosition();}
		bool checkLoadPreviusPosition(){return _oc.checkLoadPreviusPosition();}

		bool checkLoadNextIsosurface(){return _oc.checkLoadNextIsosurface();}
		bool checkLoadPreviusIsosurface(){return _oc.checkLoadPreviusIsosurface();}
	
		bool loadNextPosition();
		bool loadPreviusPosition();

		bool loadNextIsosurface();
		bool loadPreviusIsosurface();

		vmml::vector<3, int> getRealDimVolume(){ return _oc.getRealDimVolume(); }
		vmml::vector<3, int> getStartCoord(){ return _oc.getStartCoord(); }
		vmml::vector<3, int> getEndCoord() { return _oc.getEndCoord(); }
		vmml::vector<3, float> getGridStartCoord(){return _oc.getGridStartCoord();}
		vmml::vector<3, float> getGridEndCoord(){return _oc.getGridEndCoord();}
		vmml::vector<3, float> getGridRealDimVolume(){return _oc.getGridRealDimVolume();}
		int getnLevels() { return _oc.getnLevels(); }
		int getmaxLevel() { return _oc.getmaxLevel();}
		int	getCubeLevel() { return _oc.getCubeLevel(); }
		int getCubeLevelCPU(){ return _oc.getCubeLevelCPU(); } 
		int getRayCastingLevel(){ return _oc.getRayCastingLevel();}
		int	getMaxHeight(){ return _oc.getMaxHeight(); }
		float getGridMaxHeight(){ return _oc.getGridMaxHeight(); }
		int	getIsosurface(){ return _oc.getIsosurface(); }

};

}

#endif /*EQ_MIVT_OCTREE_MANAGER_H*/
