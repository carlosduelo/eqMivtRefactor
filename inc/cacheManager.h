/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CACHE_MANAGER_H
#define EQ_MIVT_CACHE_MANAGER_H

#include <controlCubeCache.h>

namespace eqMivt
{

class CacheManager
{
	private:
		ControlPlaneCache cpc;
		boost::unordered_map<device_t, ControlCubeCache *> _cubeCaches;

		int _nLevels;
		int _levelCube;
		vmml::vector<3,int> _offset;
		vmml::vector<3,int> _min;
		vmml::vector<3,int> _max;

	public:
		bool init(std::vector<std::string> parameters);

		void stop();

		bool existsDevice(device_t device);

		ControlCubeCache * getCubeCache(device_t device);

		bool freeMemoryAndPause();

		bool reSizeAndContinue(vmml::vector<3,int> min, vmml::vector<3,int> max, int nLevels, int levelCube, vmml::vector<3,int> offset);

};


}

#endif /*EQ_MIVT_CACHE_MANAGER_H*/

