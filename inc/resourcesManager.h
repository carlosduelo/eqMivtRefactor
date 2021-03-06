/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_RESOURCES_MANAGER_H
#define EQ_MIVT_RESOURCES_MANAGER_H

#include <colorManager.h>
#include <octreeManager.h>
#include <cacheManager.h>
#include <render.h>

#include <lunchbox/lock.h>

namespace eqMivt
{

class ResourcesManager
{
	private:
		OctreeManager	_oM;
		CacheManager	_cM;
		ColorManager	_coM;
		lunchbox::Lock  _lock;

		bool _isInit;

		bool _addNewDevice(Render * render);

	public:
		ResourcesManager():_isInit(false){};
		bool isInit(){ return _isInit; }

		bool init(std::vector<std::string> data_param, std::string octree_file, std::string colors_file, float memoryOccupancy);

		bool start();

		int getNumOctrees(){ return _oM.getNumOctrees(); }

		bool loadNext();
		bool loadPrevius();

		bool loadNextPosition();
		bool loadPreviusPosition();

		bool loadNextIsosurface();
		bool loadPreviusIsosurface();

		vmml::vector<3, int> getRealDimVolume(){ return _oM.getRealDimVolume(); }
		vmml::vector<3, int> getStartCoord(){ return _oM.getStartCoord(); }
		vmml::vector<3, int> getEndCoord() { return _oM.getEndCoord(); }
		vmml::vector<3, float> getGridStartCoord(){return _oM.getGridStartCoord();}
		vmml::vector<3, float> getGridEndCoord(){return _oM.getGridEndCoord();}
		vmml::vector<3, float> getGridRealDimVolume(){return _oM.getGridRealDimVolume();}
		int	getMaxHeight(){ return _oM.getMaxHeight(); }
		float getGridMaxHeight(){ return _oM.getGridMaxHeight(); }

		bool updateRender(Render * render);

		void destroy();	

};

}

#endif /* EQ_MIVT_RESOURCES_MANAGER_H */
