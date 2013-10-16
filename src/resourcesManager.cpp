/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/


#include <resourcesManager.h>

namespace eqMivt
{

bool ResourcesManager::init(std::vector<std::string> data_param, std::string octree_file, std::string colors_file)
{
	return _cM.init(data_param) &&  _oM.init(octree_file) && _coM.init(colors_file);
}

void ResourcesManager::destroy()
{
	_cM.stop();
	_oM.stop();
	_coM.destroy();
}

bool ResourcesManager::loadNextPosition()
{
	if (!_oM.checkLoadNextPosition())
		return true;

	if (_cM.freeMemoryAndPause() && _oM.loadNextPosition())
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

		if (!_cM.reSizeAndContinue(sP, eP, _oM.getnLevels(), _oM.getCubeLevel(), _oM.getStartCoord()))
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
	if (!_oM.checkLoadPreviusPosition())
		return true;

	if (_cM.freeMemoryAndPause() && _oM.loadPreviusPosition())
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

		if (!_cM.reSizeAndContinue(sP, eP, _oM.getnLevels(), _oM.getCubeLevel(), _oM.getStartCoord()))
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
	if (!_oM.checkLoadNextIsosurface())
		return true;

	return _oM.loadNextIsosurface(); 
}

bool ResourcesManager::loadPreviusIsosurface()
{
	if (!_oM.checkLoadPreviusIsosurface())
		return true;

	return _oM.loadPreviusIsosurface(); 
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
	render->setRayCastingLevel(_oM.getRayCastingLevel());
	return render->setCache(c);
}
	
bool ResourcesManager::updateRender(Render * render)
{
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
		return true;
	}
	else
	{
		return _addNewDevice(render);
	}
}

}
