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

namespace eqMivt
{

class ResourcesManager
{
	private:
		OctreeManager	_oM;
		CacheManager	_cM;
		ColorManager	_coM;

		bool _addNewDevice(Render * render);

	public:
		bool init(std::vector<std::string> data_param, std::string octree_file, std::string colors_file);

		bool loadNextPosition();
		bool loadPreviusPosition();

		bool loadNextIsosurface();
		bool loadPreviusIsosurface();

		bool updateRender(Render * render);

		void destroy();	

};

}

#endif /* EQ_MIVT_RESOURCES_MANAGER_H */
