/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <render.h>

#include <cuda_help.h>
#include <octree_cuda.h>
#include <rayCaster_cuda.h>

#ifdef TIMING
#include <lunchbox/clock.h>
#endif

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

	return true;
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
		drawCubes(_octree->getOctree(), _octree->getSizes(), _octree->getnLevels(),
								VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
								w, h, _pvpW, _pvpH, _octree->getmaxLevel(), _vC.getSizeGPU(),
								_vC.getVisibleCubesGPU(), _vC.getIndexVisibleCubesGPU(),
								VectorToInt3(_octree->getRealDim()), _octree->getGridMaxHeight(), _pixelBuffer, 
								_colors.r, _colors.g, _colors.b, 0);

		return true;

}

bool Render::_draw(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
				vmml::vector<4, float> up, vmml::vector<4, float> right,
				float w, float h)
{
	#ifdef TIMING
	lunchbox::Clock clockO;
	lunchbox::Clock clockT;
	clockT.reset();
	double oT = 0.0;
	double rT = 0.0;
	double cT = 0.0;
	double cT2 = 0.0;
	#endif

	int iterations = 0;

	while(_vC.getNumElements(PAINTED) != _vC.getSize())
	{
		#ifdef TIMING
		clockO.reset();
		#endif
		/* LAUNG OCTREE */
		_vC.updateGPU(NOCUBE, 0);

		getBoxIntersectedOctree(_octree->getOctree(), _octree->getSizes(), _octree->getnLevels(),
								VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
								w, h, _pvpW, _pvpH, _octree->getRayCastingLevel(), _vC.getSizeGPU(),
								_vC.getVisibleCubesGPU(), _vC.getIndexVisibleCubesGPU(),
								VectorToInt3(_octree->getRealDim()), 0);

		_vC.updateCPU();
		/* LAUNCH OCTREE */
		#ifdef TIMING
		oT += clockO.getTimed()/1000.0;
		clockO.reset();
		#endif

		/* Lock Cubes*/
		_cache.pushCubes(&_vC);
		_vC.updateIndexCPU();
		/* Lock Cubes*/
		#ifdef TIMING
		cT += clockO.getTimed()/1000.0;
		clockO.reset();
		#endif

		/* Ray Casting */
		_vC.updateGPU(NOCUBE | CACHED, 0);

		rayCaster(	VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
						w, h, _pvpW, _pvpH, _vC.getSizeGPU(), _octree->getRayCastingLevel(), 
						_octree->getCubeLevel(), _octree->getnLevels(), _octree->getIsosurface(),
						_vC.getVisibleCubesGPU(), _vC.getIndexVisibleCubesGPU(), 
						_octree->getGridMaxHeight(), _pixelBuffer, VectorToInt3(_octree->getRealDim()),
						_colors.r, _colors.g, _colors.b, 0);

		_vC.updateCPU();
		/* Ray Casting */
		#ifdef TIMING
		rT += clockO.getTimed()/1000.0;
		clockO.reset();
		#endif

		/* Unlock Cubes*/
		_cache.popCubes();
		/* Unlock Cubes*/
		#ifdef TIMING
		cT2 += clockO.getTimed()/1000.0;
		#endif
		
		//std::cout<<iterations<<" "<<((_vC.getSize() - _vC.getNumElements(PAINTED))*100.0f)/_vC.getSize()<<std::endl;
		iterations++;
	}
	#ifdef TIMING
	std::cout<<"===== Render Statistics ======"<<std::endl;
	std::cout<<"Time octree "<<oT<<" seconds"<<std::endl;
	std::cout<<"Time cache push "<<cT<<" seconds"<<std::endl;
	std::cout<<"Time cache pop "<<cT2<<" seconds"<<std::endl;
	std::cout<<"Time raycasting "<<rT<<" seconds"<<std::endl;
	std::cout<<"Total time for frame "<<clockT.getTimed()/1000.0<<" seconds"<<std::endl;
	std::cout<<"=============================="<<std::endl;
	#endif

	return true;
}

bool Render::frameDraw(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
				vmml::vector<4, float> up, vmml::vector<4, float> right,
				float w, float h)
{
	_vC.init();

	bool result = true;
	if (_drawCube)
	{
		result =  _drawCubes(origin, LB, up, right, w, h);
	}
	else
	{
		_cache.startFrame();
		result =  _draw(origin, LB, up, right, w, h);
		_cache.finishFrame();
	}


	return result;
}

}
