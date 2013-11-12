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

	if (cudaSuccess != cudaSetDevice(_device))
	{
		std::cerr<<"Render, cudaSetDevice error: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}

	return true;
}

void Render::destroy()
{
	workpackage_t workP;
	workP.work[0] = STOP;
	// SEND STOP
	for(int i=0; i<MAX_WORKERS; i++)
	{
		_octreeQueue[i].cond.lock();
		_octreeQueue[i].queue.push(workP);
		if (_octreeQueue[i].queue.size() == 1)
			_octreeQueue[i].cond.signal();
		_octreeQueue[i].cond.unlock();
	}
	// WAIT FOR STOP
	_masterQueue.cond.lock();
	for(int i=0; i<MAX_WORKERS; i++)
	{
		if (_masterQueue.queue.empty())
			_masterQueue.cond.wait();
		_masterQueue.queue.pop();
	}
	_masterQueue.cond.unlock();
	for(int i=0; i<MAX_WORKERS; i++)
	{
		_octreeWorkers[i].join();
		_cacheWorkers[i].join();
		_rayCasterWorkers[i].join();
		_poperWorkers[i].join();
	}
	_destroyVisibleCubes();
}

void Render::_destroyVisibleCubes()
{
	if (_visibleCubes != 0)
		delete[] _visibleCubes;
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

	if (_octree != 0 && _colors.r != 0 && _ccc != 0)
	{
		for(int i=0; i<MAX_WORKERS; i++)
		{
			_octreeWorkers[i].init(_octree, _device, &_octreeQueue[i], &_cacheQueue[i], &_parameters);
			_cacheWorkers[i].init(_octree, _ccc, _device, &_cacheQueue[i], &_rayCasterQueue[i], &_parameters);
			_rayCasterWorkers[i].init(_octree, _pixelBuffer, &_colors, _device, &_rayCasterQueue[i], &_octreeQueue[i], &_parameters, &_masterQueue, &_popQueue[i]);
			_poperWorkers[i].init(&_popQueue[i], _cacheWorkers[i].getCache());
		}
		for(int i=0; i<MAX_WORKERS; i++)
		{
			_octreeWorkers[i].start();
			_cacheWorkers[i].start();
			_rayCasterWorkers[i].start();
			_poperWorkers[i].start();
		}
	}

	return true;
}

void Render::setOctree(Octree * octree)
{ 
	_octree = octree;
	if (_octree != 0 && _colors.r != 0 && _ccc != 0)
	{
		for(int i=0; i<MAX_WORKERS; i++)
		{
			_octreeWorkers[i].init(_octree, _device, &_octreeQueue[i], &_cacheQueue[i], &_parameters);
			_cacheWorkers[i].init(_octree, _ccc, _device, &_cacheQueue[i], &_rayCasterQueue[i], &_parameters);
			_rayCasterWorkers[i].init(_octree, _pixelBuffer, &_colors, _device, &_rayCasterQueue[i], &_octreeQueue[i], &_parameters, &_masterQueue, &_popQueue[i]);
			_poperWorkers[i].init(&_popQueue[i], _cacheWorkers[i].getCache());
		}
		for(int i=0; i<MAX_WORKERS; i++)
		{
			_octreeWorkers[i].start();
			_cacheWorkers[i].start();
			_rayCasterWorkers[i].start();
			_poperWorkers[i].start();
		}
	}
}

void Render::setColors(color_t colors)
{ 
	_colors = colors;
	if (_octree != 0 && _colors.r != 0 && _ccc != 0)
	{
		for(int i=0; i<MAX_WORKERS; i++)
		{
			_octreeWorkers[i].init(_octree, _device, &_octreeQueue[i], &_cacheQueue[i], &_parameters);
			_cacheWorkers[i].init(_octree, _ccc, _device, &_cacheQueue[i], &_rayCasterQueue[i], &_parameters);
			_rayCasterWorkers[i].init(_octree, _pixelBuffer, &_colors, _device, &_rayCasterQueue[i], &_octreeQueue[i], &_parameters, &_masterQueue, &_popQueue[i]);
			_poperWorkers[i].init(&_popQueue[i], _cacheWorkers[i].getCache());
		}
		for(int i=0; i<MAX_WORKERS; i++)
		{
			_octreeWorkers[i].start();
			_cacheWorkers[i].start();
			_rayCasterWorkers[i].start();
			_poperWorkers[i].start();
		}
	}
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
			_visibleCubes = new visibleCube_t[_size];

			if (cudaSuccess != cudaMalloc((void**)&_visibleCubesGPU, _size*sizeof(visibleCube_t)))
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
		drawCubes(	_octree->getOctree(), _octree->getSizes(), _octree->getnLevels(),
					VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
					w, h, _pvpW, _pvpH, _octree->getmaxLevel(), _size,
					VectorToInt3(_octree->getRealDim()), _octree->getGridMaxHeight(), 
					_colors.r, _colors.g, _colors.b, _pixelBuffer, 0);
		return true;
}

bool Render::_draw(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
				vmml::vector<4, float> up, vmml::vector<4, float> right,
				float w, float h)
{
	/* SET SHARED PARAMETERS */
	_parameters.size = _pvpW*_pvpH;
	_parameters.visibleCubes = _visibleCubes;
	_parameters.visibleCubesGPU = _visibleCubesGPU;
	_parameters.pvpW = _pvpW;
	_parameters.pvpH = _pvpH;
	_parameters.w = w;
	_parameters.h = h;
	_parameters.origin = VectorToFloat3(origin);
	_parameters.LB = VectorToFloat3(LB);
	_parameters.up = VectorToFloat3(up);
	_parameters.right = VectorToFloat3(right);
	/* END SET SHARED PARAMETERS*/
	
	workpackage_t workP;
	workP.work[0] = START_FRAME;
	/* SEND START FRAME */
	for(int i=0; i<MAX_WORKERS; i++)
	{
		_octreeQueue[i].cond.lock();
		_octreeQueue[i].queue.push(workP);
		if (_octreeQueue[i].queue.size() == 1)
			_octreeQueue[i].cond.signal();
		_octreeQueue[i].cond.unlock();
	}

	/* SEND FRAME */
	workP.work[0] = FRAME;
	workP.work[1] = 0;
	workP.work[2] = _pvpW*_pvpH;
	for(int i=0; i<MAX_WORKERS; i++)
	{
		_octreeQueue[i].cond.lock();
		_octreeQueue[i].queue.push(workP);
		if (_octreeQueue[i].queue.size() == 1)
			_octreeQueue[i].cond.signal();
		_octreeQueue[i].cond.unlock();
	}

	// WAIT FOR FRAMES
	_masterQueue.cond.lock();
	for(int i=0; i<MAX_WORKERS; i++)
	{
		if (_masterQueue.queue.empty())
			_masterQueue.cond.wait();
		workpackage_t p = _masterQueue.queue.front();
		std::cout<<p.work[0]<<" "<<p.work[1]<<" "<<p.work[2]<<std::endl;
		_masterQueue.queue.pop();
	}


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
		if (cudaSuccess != cudaMemset((void**)_visibleCubesGPU, 0, _size*sizeof(visibleCube_t)))
		{
			std::cerr<<"Render, error int memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return false;
		}
		result =  _draw(origin, LB, up, right, w, h);
	}

	return result;
}

}
