/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <cacheManager.h>

namespace eqMivt
{

bool CacheManager::init(std::vector<std::string> parameters, float memoryOccupancy)
{
	return cccCPU.initParameter(parameters, memoryOccupancy);
}

void CacheManager::stop()
{
	boost::unordered_map<device_t, ControlCubeCache *>::iterator it = _cubeCaches.begin();
	while(it != _cubeCaches.end())
	{
		it->second->stopCache();
		delete it->second;
		it++;
	}
	_cubeCaches.clear();
	cccCPU.stopCache();
}

bool CacheManager::reSizeAndContinue(vmml::vector<3,int> min, vmml::vector<3,int> max, int nLevels, int levelCubeCPU, int levelCube, vmml::vector<3,int> offset)
{
	_min = min;
	_max = max;
	_nLevels = nLevels;
	_levelCube = levelCube;
	_offset = offset;

	bool result = true;
	boost::unordered_map<device_t, ControlCubeCache *>::iterator it = _cubeCaches.begin();
	while(it != _cubeCaches.end())
	{
		result = it->second->reSizeCacheAndContinue(_nLevels, _levelCube, _offset);
		if (!result)
			return false;
		it++;
	}
	return cccCPU.reSizeCacheAndContinue(_min, _max, levelCubeCPU, nLevels);
}

bool CacheManager::existsDevice(device_t device)
{
	boost::unordered_map<device_t, ControlCubeCache *>::iterator it = _cubeCaches.find(device);
	return it != _cubeCaches.end();
}

ControlCubeCache * CacheManager::getCubeCache(device_t device)
{
	boost::unordered_map<device_t, ControlCubeCache *>::iterator it = _cubeCaches.find(device);
	if (it != _cubeCaches.end())
		return it->second;

	ControlCubeCache * c = new ControlCubeCache();
	bool r =	c->initParameter(&cccCPU, device) &&
				c->freeCacheAndPause() && 
				c->reSizeCacheAndContinue(_nLevels, _levelCube, _offset);
	
	if (r)
		_cubeCaches.insert(std::make_pair<device_t, ControlCubeCache *>(device, c));
	else
		return 0;

	return c;
}
	
bool CacheManager::freeMemoryAndPause()
{
	boost::unordered_map<device_t, ControlCubeCache *>::iterator it = _cubeCaches.begin();
	while(it != _cubeCaches.end())
	{
		if (!it->second->freeCacheAndPause())
			return false;
		it++;
	}
	return cccCPU.freeCacheAndPause();
}
}
