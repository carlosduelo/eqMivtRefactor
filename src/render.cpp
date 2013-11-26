/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <render.h>

#include <octree_cuda.h>

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
	_size = 0;
	_visibleCubes = 0;
	_visibleCubesGPU = 0;
	_ccc = 0;
	_pvpH = 0;
	_pvpW = 0;

	if (cudaSuccess != cudaSetDevice(_device))
	{
		std::cerr<<"Render, cudaSetDevice error: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}
	if (cudaSuccess != cudaStreamCreate(&_stream))
	{
		std::cerr<<"Render, cuda create stream error: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}
	return true;
}

void Render::destroy()
{
	if (cudaSuccess != cudaStreamDestroy(_stream))
	{
		std::cerr<<"Render, cuda create stream error: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return;
	}

	_destroyVisibleCubes();
	_cache.destroy();
}

void Render::_destroyVisibleCubes()
{
	if (_visibleCubes != 0)
		if (cudaSuccess != cudaFreeHost(_visibleCubes))
		{
			std::cerr<<"Render, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return;
		}
	if (_visibleCubesGPU != 0)
		if (cudaSuccess != cudaFree(_visibleCubesGPU))
		{
			std::cerr<<"Render, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return;
		}
}

bool Render::setCache(ControlCubeCache * ccc)
{
	_ccc = ccc;	

	return _cache.init(_ccc);
}

void Render::setOctree(Octree * octree)
{ 
	_octree = octree;
}

void Render::setColors(color_t colors)
{ 
	_colors = colors;
}

bool Render::setViewPort(int pvpW, int pvpH)
{
	if (pvpW != _pvpW || pvpH != _pvpH)
	{
		_pvpW = pvpW;
		_pvpH = pvpH;
		if (_size != _pvpW*_pvpH)
		{
			_size = _pvpW*_pvpH;
			_destroyVisibleCubes();
			if (cudaSuccess != cudaHostAlloc((void**)&_visibleCubes, _size*sizeof(visibleCube_t), cudaHostAllocDefault) ||
				cudaSuccess != cudaMalloc((void**)&_visibleCubesGPU, _size*sizeof(visibleCube_t)))
			{
				std::cerr<<"Render, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
				return false;
			}
		}
	}

	return true;
}

bool Render::_drawCubes(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
				vmml::vector<4, float> up, vmml::vector<4, float> right,
				float w, float h)
{
	#ifdef TIMING
	lunchbox::Clock clock;
	clock.reset();
	#endif
		drawCubes(	_octree->getOctree(), _octree->getSizes(), _octree->getnLevels(),
					VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
					w, h, _pvpW, _pvpH, _octree->getmaxLevel(), _size,
					VectorToInt3(_octree->getRealDim()), _octree->getGridMaxHeight(), 
					_colors.r, _colors.g, _colors.b, _pixelBuffer, 0);
	#ifdef TIMING
	std::cout<<"Time render a frame "<<clock.getTimed()/1000.0f<<" seconds"<<std::endl;
	#endif
		return true;
}

bool Render::_draw(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
				vmml::vector<4, float> up, vmml::vector<4, float> right,
				float w, float h)
{
	#ifdef TIMING
	lunchbox::Clock clock;
	clock.reset();
	#endif

	bool newIteration = true;
	float3 originC = VectorToFloat3(origin);
	float3 LBC = VectorToFloat3(LB);
	float3 upC = VectorToFloat3(up);
	float3 rightC = VectorToFloat3(right);
	int3 realDimC = VectorToInt3(_octree->getRealDim());

	int iterations = 0;
	while(newIteration)
	{
		float ** tCubes = _cache.syncAndGetTableCubes(_stream);

		getBoxIntersectedOctree(_octree->getOctree(), _octree->getSizes(), _octree->getnLevels(),
								originC, LBC, upC, rightC,
								w, h, _pvpW, _pvpH, _octree->getRayCastingLevel(), _octree->getCubeLevel(), _size, 
								_visibleCubesGPU, 0, realDimC, 
								_colors.r, _colors.g, _colors.b, _pixelBuffer, 
								_octree->getIsosurface(), _octree->getGridMaxHeight(), 
								tCubes, _stream);

		if (cudaSuccess != cudaMemcpyAsync((void*)(_visibleCubes), (void*)(_visibleCubesGPU), _size*sizeof(visibleCube_t), cudaMemcpyDeviceToHost, _stream))
		{
			std::cerr<<"Visible cubes, error updating cpu copy: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return false;
		}
		if (cudaSuccess != cudaStreamSynchronize(_stream))
		{
			std::cerr<<"Error sync: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return false;
		}

		int cF = 0;
		for(int i=0; i<_size; i++)
		{
			if (_visibleCubes[i].idCube != 0)
				_cache.popCubes(_visibleCubes[i].idCube);

			_cache.pushCubes(&_visibleCubes[i]);

			if(_visibleCubes[i].state == PAINTED)
			{
				cF++;
			}
		}

		if (cF == _size)
		{
			newIteration = false;
		}
		else
		{
			if (cudaSuccess != cudaMemcpyAsync((void*)(_visibleCubesGPU), (void*)(_visibleCubes), _size*sizeof(visibleCube_t), cudaMemcpyHostToDevice, _stream))
			{
				std::cerr<<"Visible cubes, error updating gpu copy: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
				return false;
			}
		}
		iterations++;
	}

	#ifdef TIMING
	std::cout<<"Time render a frame "<<clock.getTimed()/1000.0f<<" seconds"<<std::endl;
	#endif

	return true;
}

bool Render::frameDraw(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
				vmml::vector<4, float> up, vmml::vector<4, float> right,
				float w, float h)
{
	bool result = true;
	if (_drawCube)
	{
		result =  _drawCubes(origin, LB, up, right, w, h);
	}
	else
	{
		_cache.setRayCastingLevel(_octree->getRayCastingLevel());
		_cache.startFrame();
		if (cudaSuccess != cudaMemset((void**)_visibleCubesGPU, 0, _size*sizeof(visibleCube_t)))
		{
			std::cerr<<"Render, error int memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return false;
		}
		result =  _draw(origin, LB, up, right, w, h);
		_cache.finishFrame();
	}

	return result;
}

}
