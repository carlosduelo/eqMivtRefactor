/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/


#include <octreeManager.h>

namespace eqMivt
{
	
bool OctreeManager::init(std::string file_name)
{
	return _oc.init(file_name);
}

void OctreeManager::stop()
{
	boost::unordered_map<device_t, Octree *>::iterator it = _octrees.begin();
	while(it != _octrees.end())
	{
		it->second->stop();
		delete it->second;
		it++;
	}
	_octrees.clear();
	_oc.stop();
}

int OctreeManager::getCurrentOctree()
{
	return _oc.getCurrentOctree();
}

bool OctreeManager::existsDevice(device_t device)
{
	boost::unordered_map<device_t, Octree *>::iterator it = _octrees.find(device);
	return it != _octrees.end();
}

Octree * OctreeManager::getOctree(device_t device)
{
	boost::unordered_map<device_t, Octree *>::iterator it = _octrees.find(device);
	if (it != _octrees.end())
		return it->second;
	
	Octree * o = new Octree();
	bool r = o->init(&_oc, device) && o->loadCurrentOctree();

	if (r)
		_octrees.insert(std::make_pair<device_t, Octree *>(device, o));
	else
		return 0;

	return o;
}

bool OctreeManager::loadNextPosition()
{
	if (_oc.loadNextPosition())
	{
		boost::unordered_map<device_t, Octree *>::iterator it = _octrees.begin();
		while(it != _octrees.end())
		{
			if (!it->second->loadCurrentOctree())
				return false;
			it++;
		}
		return true;
	}
	else
		return false;
}

bool OctreeManager::loadPreviusPosition()
{
	if (_oc.loadPreviusPosition())
	{
		boost::unordered_map<device_t, Octree *>::iterator it = _octrees.begin();
		while(it != _octrees.end())
		{
			if (!it->second->loadCurrentOctree())
				return false;
			it++;
		}
		return true;
	}
	else
		return false;
}

bool OctreeManager::loadNextIsosurface()
{
	if (_oc.loadNextIsosurface())
	{
		boost::unordered_map<device_t, Octree *>::iterator it = _octrees.begin();
		while(it != _octrees.end())
		{
			if (!it->second->loadCurrentOctree())
				return false;
			it++;
		}
		return true;
	}
	else
		return false;
}

bool OctreeManager::loadPreviusIsosurface()
{
	if (_oc.loadPreviusIsosurface())
	{
		boost::unordered_map<device_t, Octree *>::iterator it = _octrees.begin();
		while(it != _octrees.end())
		{
			if (!it->second->loadCurrentOctree())
				return false;
			it++;
		}
		return true;
	}
	else
		return false;
}

}
