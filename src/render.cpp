/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <render.h>

#include <cuda_help.h>
#include <octree_cuda.h>
#include <rayCaster_cuda.h>

namespace eqMivt
{

bool Render::init(device_t device)
{
	_device = device;
	_octree = 0;
	_colors.r = 0;
	_colors.g = 0;
	_colors.b = 0;
	_pixelBuffer = 0;
}

void Render::destroy()
{
	_vC.destroy();
}

bool Render::setCache(ControlCubeCache * ccc)
{
	return _cache.init(ccc);
}

bool Render::setViewPort(int pvpW, int pvpH)
{
	if (pvpW != _pvpW || pvpH != _pvpH)
	{
		_pvpW = pvpW;
		_pvpH = pvpH;
		_vC.reSize(_pvpW*_pvpH); 	
	}

	return true;
}

bool Render::_drawCubes(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
				vmml::vector<4, float> up, vmml::vector<4, float> right,
				float w, float h)
{
		getBoxIntersectedOctree(_octree->getOctree(), _octree->getSizes(), _octree->getnLevels(),
								VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
								w, h, _pvpW, _pvpH, _octree->getmaxLevel(), _vC.getSizeGPU(),
								_vC.getVisibleCubesGPU(), _vC.getIndexVisibleCubesGPU(),
								_octree->getxGrid(), _octree->getyGrid(), _octree->getzGrid(),
								VectorToInt3(_octree->getRealDim()), 0);

		rayCasterCubes(	VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
						w, h, _pvpW, _pvpH, _vC.getSizeGPU(), _octree->getmaxLevel(), _octree->getnLevels(),
						_vC.getVisibleCubesGPU(), _vC.getIndexVisibleCubesGPU(), 
						_octree->getMaxHeight(), _pixelBuffer, _octree->getxGrid(), _octree->getyGrid(), _octree->getzGrid(),
						VectorToInt3(_octree->getRealDim()), _colors.r, _colors.g, _colors.b, 0);

		return true;

}

bool Render::_draw(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
				vmml::vector<4, float> up, vmml::vector<4, float> right,
				float w, float h)
{
	while(_vC.getNumElements(PAINTED) != _vC.getSize())
	{
		/* LAUNG OCTREE */
		_vC.updateGPU(NOCUBE, 0);

		getBoxIntersectedOctree(_octree->getOctree(), _octree->getSizes(), _octree->getnLevels(),
								VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
								w, h, _pvpW, _pvpH, _octree->getRayCastingLevel(), _vC.getSizeGPU(),
								_vC.getVisibleCubesGPU(), _vC.getIndexVisibleCubesGPU(),
								_octree->getxGrid(), _octree->getyGrid(), _octree->getzGrid(),
								VectorToInt3(_octree->getRealDim()), 0);

		_vC.updateCPU();
		/* LAUNCH OCTREE */

		/* Lock Cubes*/
		_cache.pushCubes(&_vC);
		_vC.updateIndexCPU();
		/* Lock Cubes*/

		/* Ray Casting */
		_vC.updateGPU(NOCUBE | CACHED, 0);

		rayCaster(	VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
						w, h, _pvpW, _pvpH, _vC.getSizeGPU(), _octree->getRayCastingLevel(), 
						_octree->getCubeLevel(), _octree->getnLevels(), _octree->getIsosurface(),
						_vC.getVisibleCubesGPU(), _vC.getIndexVisibleCubesGPU(), 
						_octree->getMaxHeight(), _pixelBuffer, _octree->getxGrid(), _octree->getyGrid(), _octree->getzGrid(),
						VectorToInt3(_octree->getRealDim()), _colors.r, _colors.g, _colors.b, 0);

		_vC.updateCPU();
		/* Ray Casting */

		/* Unlock Cubes*/
		_cache.popCubes();
		/* Unlock Cubes*/
	}
}

bool Render::frameDraw(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
				vmml::vector<4, float> up, vmml::vector<4, float> right,
				float w, float h)
{
	_vC.init();

	if (_drawCube)
		return _drawCubes(origin, LB, up, right, w, h);
	else
		return _draw(origin, LB, up, right, w, h);
}

}
