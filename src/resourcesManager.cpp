/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/


#include <resourcesManager.h>

namespace eqMivt
{

bool ResourcesManager::init(std::vector<std::string> data_param, std::string octree_file, std::string colors_file, float memoryOccupancy)
{
	return _cM.init(data_param, memoryOccupancy) &&  _oM.init(octree_file) && _coM.init(colors_file);
}

bool ResourcesManager::start()
{
		vmml::vector<3, int> realDimVolume = _oM.getRealDimVolume();
		// Resize cube cache and plane cache
		vmml::vector<3, int> sP, eP;
		sP[0] = _oM.getStartCoord().x() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().x() - CUBE_INC; 
		sP[1] = _oM.getStartCoord().y() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().y() - CUBE_INC; 
		sP[2] = _oM.getStartCoord().z() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().z() - CUBE_INC; 
		eP[0] = _oM.getEndCoord().x() + CUBE_INC >= realDimVolume.x() ? realDimVolume.x() : _oM.getEndCoord().x() + CUBE_INC;
		eP[1] = _oM.getEndCoord().y() + CUBE_INC >= realDimVolume.y() ? realDimVolume.y() : _oM.getEndCoord().y() + CUBE_INC;
		eP[2] = _oM.getEndCoord().z() + CUBE_INC >= realDimVolume.z() ? realDimVolume.z() : _oM.getEndCoord().z() + CUBE_INC;

		if (!_cM.freeMemoryAndPause() || !_cM.reSizeAndContinue(sP, eP, _oM.getnLevels(), _oM.getCubeLevelCPU(), _oM.getCubeLevel(), _oM.getStartCoord()))
		{
			std::cerr<<"Error, resizing plane cache"<<std::endl;
			return false;
		}

	return true;
}

void ResourcesManager::destroy()
{
	_cM.stop();
	_oM.stop();
	_coM.destroy();
}

bool ResourcesManager::loadNext()
{
	if (_oM.checkLoadNextIsosurface())
	{
		loadNextIsosurface();
		return true;
	}
	else
	{
		if (_oM.checkLoadNextPosition())
		{
			loadNextPosition();
			return true;
		}
		else
		{
			return false;
		}
	}

	return false;
}

bool ResourcesManager::loadPrevius()
{
	if (_oM.checkLoadPreviusIsosurface())
	{
		loadPreviusIsosurface();
		return true;
	}
	else
	{
		if (_oM.checkLoadPreviusPosition())
		{
			loadPreviusPosition();
			return true;
		}
		else
		{
			return false;
		}
	}
}

bool ResourcesManager::loadNextPosition()
{
	if (_oM.checkLoadNextPosition() && _cM.freeMemoryAndPause() && _oM.loadNextPosition())
	{
		vmml::vector<3, int> realDimVolume = _oM.getRealDimVolume();
		// Resize cube cache and plane cache
		vmml::vector<3, int> sP, eP;
		sP[0] = _oM.getStartCoord().x() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().x() - CUBE_INC; 
		sP[1] = _oM.getStartCoord().y() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().y() - CUBE_INC; 
		sP[2] = _oM.getStartCoord().z() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().z() - CUBE_INC; 
		eP[0] = _oM.getEndCoord().x() + CUBE_INC >= realDimVolume.x() ? realDimVolume.x() : _oM.getEndCoord().x() + CUBE_INC;
		eP[1] = _oM.getEndCoord().y() + CUBE_INC >= realDimVolume.y() ? realDimVolume.y() : _oM.getEndCoord().y() + CUBE_INC;
		eP[2] = _oM.getEndCoord().z() + CUBE_INC >= realDimVolume.z() ? realDimVolume.z() : _oM.getEndCoord().z() + CUBE_INC;

		if (!_cM.reSizeAndContinue(sP, eP, _oM.getnLevels(), _oM.getCubeLevelCPU(), _oM.getCubeLevel(), _oM.getStartCoord()))
		{
			std::cerr<<"Error, resizing plane cache"<<std::endl;
			return false;
		}
	}
	else
		return false;

	return true;
}

bool ResourcesManager::loadPreviusPosition()
{
	if (_oM.checkLoadPreviusPosition() && _cM.freeMemoryAndPause() && _oM.loadPreviusPosition())
	{
		vmml::vector<3, int> realDimVolume = _oM.getRealDimVolume();
		// Resize cube cache and plane cache
		vmml::vector<3, int> sP, eP;
		sP[0] = _oM.getStartCoord().x() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().x() - CUBE_INC; 
		sP[1] = _oM.getStartCoord().y() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().y() - CUBE_INC; 
		sP[2] = _oM.getStartCoord().z() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().z() - CUBE_INC; 
		eP[0] = _oM.getEndCoord().x() + CUBE_INC >= realDimVolume.x() ? realDimVolume.x() : _oM.getEndCoord().x() + CUBE_INC;
		eP[1] = _oM.getEndCoord().y() + CUBE_INC >= realDimVolume.y() ? realDimVolume.y() : _oM.getEndCoord().y() + CUBE_INC;
		eP[2] = _oM.getEndCoord().z() + CUBE_INC >= realDimVolume.z() ? realDimVolume.z() : _oM.getEndCoord().z() + CUBE_INC;

		if (!_cM.reSizeAndContinue(sP, eP, _oM.getnLevels(), _oM.getCubeLevelCPU(), _oM.getCubeLevel(), _oM.getStartCoord()))
		{
			std::cerr<<"Error, resizing plane cache"<<std::endl;
			return false;
		}
	}
	else
		return false;

	return true;
}

bool ResourcesManager::loadNextIsosurface()
{
	if (_oM.checkLoadNextIsosurface() && _cM.freeMemoryAndPause() && _oM.loadNextIsosurface())
	{
		vmml::vector<3, int> realDimVolume = _oM.getRealDimVolume();
		// Resize cube cache and plane cache
		vmml::vector<3, int> sP, eP;
		sP[0] = _oM.getStartCoord().x() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().x() - CUBE_INC; 
		sP[1] = _oM.getStartCoord().y() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().y() - CUBE_INC; 
		sP[2] = _oM.getStartCoord().z() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().z() - CUBE_INC; 
		eP[0] = _oM.getEndCoord().x() + CUBE_INC >= realDimVolume.x() ? realDimVolume.x() : _oM.getEndCoord().x() + CUBE_INC;
		eP[1] = _oM.getEndCoord().y() + CUBE_INC >= realDimVolume.y() ? realDimVolume.y() : _oM.getEndCoord().y() + CUBE_INC;
		eP[2] = _oM.getEndCoord().z() + CUBE_INC >= realDimVolume.z() ? realDimVolume.z() : _oM.getEndCoord().z() + CUBE_INC;

		if (!_cM.reSizeAndContinue(sP, eP, _oM.getnLevels(), _oM.getCubeLevelCPU(), _oM.getCubeLevel(), _oM.getStartCoord()))
		{
			std::cerr<<"Error, resizing plane cache"<<std::endl;
			return false;
		}
	}
	else
		return false;

	return true;
}

bool ResourcesManager::loadPreviusIsosurface()
{
	if (_oM.checkLoadPreviusIsosurface() && _cM.freeMemoryAndPause() && _oM.loadPreviusIsosurface())
	{
		vmml::vector<3, int> realDimVolume = _oM.getRealDimVolume();
		// Resize cube cache and plane cache
		vmml::vector<3, int> sP, eP;
		sP[0] = _oM.getStartCoord().x() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().x() - CUBE_INC; 
		sP[1] = _oM.getStartCoord().y() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().y() - CUBE_INC; 
		sP[2] = _oM.getStartCoord().z() - CUBE_INC < 0 ? 0 : _oM.getStartCoord().z() - CUBE_INC; 
		eP[0] = _oM.getEndCoord().x() + CUBE_INC >= realDimVolume.x() ? realDimVolume.x() : _oM.getEndCoord().x() + CUBE_INC;
		eP[1] = _oM.getEndCoord().y() + CUBE_INC >= realDimVolume.y() ? realDimVolume.y() : _oM.getEndCoord().y() + CUBE_INC;
		eP[2] = _oM.getEndCoord().z() + CUBE_INC >= realDimVolume.z() ? realDimVolume.z() : _oM.getEndCoord().z() + CUBE_INC;

		if (!_cM.reSizeAndContinue(sP, eP, _oM.getnLevels(), _oM.getCubeLevelCPU(), _oM.getCubeLevel(), _oM.getStartCoord()))
		{
			std::cerr<<"Error, resizing plane cache"<<std::endl;
			return false;
		}
	}
	else
		return false;

	return true;
}

bool ResourcesManager::_addNewDevice(Render * render)
{
	Octree * o = _oM.getOctree(render->getDevice());
	color_t co = _coM.getColors(render->getDevice());
	ControlCubeCache * c = _cM.getCubeCache(render->getDevice());

	if (o == 0 || c == 0)
		return false;

	render->setOctree(o);
	render->setColors(co);
	render->setCache(c);
	render->setRayCastingLevel(_oM.getRayCastingLevel());
	return true;
}
	
bool ResourcesManager::updateRender(Render * render)
{
	_lock.set();

	#ifndef NDEBUG
	if (_cM.existsDevice(render->getDevice()) != _oM.existsDevice(render->getDevice()))
	{
		std::cerr<<"Inconsistent resources"<<std::endl;
		throw;
	}
	#endif

	if (render->cacheIsInit() && render->octreeIsInit() && render->colorsIsInit())
	{
		render->setRayCastingLevel(_oM.getRayCastingLevel());
		_lock.unset();
		return true;
	}
	else
	{
		bool r = _addNewDevice(render);
		_lock.unset();
		return r;
	}
}

}
