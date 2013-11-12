/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <renderWorkers.h>

#include <octree_cuda.h>
#include <rayCaster_cuda.h>

namespace eqMivt
{

bool worker::_init(Octree * octree, device_t device, queue_t * IN, queue_t * OUT, sharedParameters_t *	parameters)
{
	_octree = octree;
	_parameters = parameters;
	_IN = IN;
	_OUT = OUT;
	_device = device;

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

bool worker::destroy()
{
	if (cudaSuccess != cudaStreamDestroy(_stream))
	{
		std::cerr<<"Render, cuda create stream error: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}

	return true;
}

bool worker::_updateCPU(int offset, int size)
{
	if (cudaSuccess != cudaMemcpyAsync((void*)(_parameters->visibleCubes+offset), (void*)(_parameters->visibleCubesGPU+offset), size*sizeof(visibleCube_t), cudaMemcpyDeviceToHost, _stream))
	{
		std::cerr<<"Visible cubes, error updating cpu copy: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}
	return true;
}

bool worker::_updateGPU(int offset, int size)
{
	if (cudaSuccess != cudaMemcpyAsync((void*)(_parameters->visibleCubesGPU+offset), (void*)(_parameters->visibleCubes+offset), size*sizeof(visibleCube_t), cudaMemcpyDeviceToHost, _stream))
	{
		std::cerr<<"Visible cubes, error updating cpu copy: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}
	return true;
}

bool worker::_sync()
{
	if (cudaSuccess != cudaStreamSynchronize(_stream))
	{
		std::cerr<<"Error octree: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}
	return true;
}

void worker::run()
{
	bool notEnd = true;
	while(notEnd)
	{
		_IN->cond.lock();
		if (_IN->queue.empty())
			_IN->cond.wait();
		workpackage_t workP = _IN->queue.front();
		_IN->queue.pop();
		_IN->cond.unlock();

		bool resend = false;

		switch(workP.work[0])
		{
			case START_FRAME: 
				resend = _startFrame();	
				break;
			case FINISH_FRAME:
				resend = _finishFrame();
				break;
			case STOP:
				notEnd = false;
				resend = _stopWorking();
				break;
			case FRAME:
				resend = _frame(workP);
				break;
			default: 
				std::cerr<<"Operation not allowed"<<std::endl;
				break;
		}
		
		if (resend)
		{
			_OUT->cond.lock();
			_OUT->queue.push(workP);
			if (_OUT->queue.size() == 1)
				_OUT->cond.signal();
			_OUT->cond.unlock();
		}
	}
}


bool workerOctree::init(Octree * octree, device_t device, queue_t * IN, queue_t * OUT, sharedParameters_t *	parameters)
{
	return _init(octree, device, IN, OUT, parameters);
}

bool workerOctree::_frame(workpackage_t work)
{
	getBoxIntersectedOctree(_octree->getOctree(), _octree->getSizes(), _octree->getnLevels(),
							_parameters->origin, _parameters->LB, _parameters->up, _parameters->right,
							_parameters->w, _parameters->h, _parameters->pvpW, _parameters->pvpH,
							_octree->getRayCastingLevel(), work.work[2], _parameters->visibleCubesGPU + work.work[1],
							(VectorToInt3(_octree->getRealDim())), 0);

	_updateCPU(work.work[1], work.work[2]);
	_sync();
	return true;
}

bool  workerOctree::_startFrame()
{
std::cout<<"Octree start frame"<<std::endl;
	return true;
}

bool workerOctree::_finishFrame()
{
std::cout<<"Octree end frame"<<std::endl;
	return true;
}

bool workerOctree::_stopWorking()
{
	return true;
}

bool workerCache::init(Octree * octree, ControlCubeCache * ccc, device_t device, queue_t * IN, queue_t * OUT, sharedParameters_t *	parameters)
{
	return _cache.init(ccc) && _init(octree, device, IN, OUT, parameters);
}

bool workerCache::_frame(workpackage_t work)
{
	visibleCube_t * cubes = _parameters->visibleCubes + work.work[1];
	for(int i=0; i<work.work[2]; i++)
	{
		_cache.pushCubes(cubes + i);
	}
	_updateGPU(work.work[1], work.work[2]);
	_sync();
	return true;
}

bool workerCache::_startFrame()
{
	_cache.setRayCastingLevel(_octree->getRayCastingLevel());
	_cache.startFrame();
std::cout<<"Cache start frame"<<std::endl;
	return true;
}

bool workerCache::_finishFrame()
{
	_cache.finishFrame();
std::cout<<"Cacheend frame"<<std::endl;
	return true;
}
bool workerCache::_stopWorking()
{
	return true;
}

bool workerRayCaster::init(Octree * octree, float * screen, color_t * colors, device_t device, queue_t * IN, queue_t * OUT, sharedParameters_t * parameters, queue_t * END, queuePOP_t * POP)
{
	_POP = POP;
	_END = END;
	_screen = screen;
	_colors = colors;
	return _init(octree, device, IN, OUT, parameters);
}

bool workerRayCaster::_frame(workpackage_t work)
{
	rayCaster(	_parameters->origin, _parameters->LB, _parameters->up, _parameters->right,
				_parameters->w, _parameters->h, _parameters->pvpW, _parameters->pvpH, 
				work.work[2], _octree->getRayCastingLevel(), 
				_octree->getCubeLevel(), _octree->getnLevels(), _octree->getIsosurface(),
				_parameters->visibleCubesGPU + work.work[1], _octree->getGridMaxHeight(), _screen + work.work[1], 
				VectorToInt3(_octree->getRealDim()),
				_colors->r, _colors->g, _colors->b, 0);

	_updateCPU(work.work[1], work.work[2]);
	_sync();

	bool sendO = true;
	int cF = 0;
	visibleCube_t * cubes = _parameters->visibleCubes + work.work[1];
	for(int i=0; i<work.work[2]; i++)
	{
		if (sendO && cubes[i].state == NOCUBE)
		{
			sendO = false;	
		}
		else if(cubes[i].state == DONE)
		{
			cF++;
		}
		else if (cubes[i].state == PAINTED)
		{
			cF++;
			_POP->cond.lock();
			_POP->queue.push(cubes[i].id);
			if (_POP->queue.size() == 1)
				_POP->cond.signal();
			_POP->cond.unlock();
			cubes[i].state = DONE;
		}
		#ifndef NDEBUG
		else
		{
			std::cerr<<"Error operation not allowed"<<std::endl;
			throw;
		}
		#endif
	}

	if (cF == work.work[2])
	{
		#ifndef NDEBUG
		if (!sendO)
		{
			std::cerr<<"Error operation not allowed"<<std::endl;
			throw;
		}
		#endif
		_END->cond.lock();
		_END->queue.push(work);
		if (_END->queue.size() == 1)
			_END->cond.signal();
		_END->cond.unlock();

	}

	return !sendO; 
}

bool workerRayCaster::_startFrame()
{
std::cout<<"ray caster start frame"<<std::endl;
	return false;
}

bool workerRayCaster::_finishFrame()
{
std::cout<<"ray caster end frame"<<std::endl;
	return false;
}

bool workerRayCaster::_stopWorking()
{
	_POP->cond.lock();
	_POP->queue.push(0);
	if (_POP->queue.size() == 1)
		_POP->cond.signal();
	_POP->cond.unlock();

	workpackage_t w; w.work[0] = STOP;
	_END->cond.lock();
	_END->queue.push(w);
	if (_END->queue.size() == 1)
		_END->cond.signal();
	_END->cond.unlock();

	return false;
}

bool workerPoper::init(queuePOP_t * POP, Cache * cache)
{
	_cache = cache;
	_POP = POP;
	return true;
}

void workerPoper::run()
{
	bool notEnd = true;
	while(notEnd)
	{
		_POP->cond.lock();
		if (_POP->queue.empty())
			_POP->cond.wait();
		index_node_t id = _POP->queue.front();
		_POP->queue.pop();
		_POP->cond.unlock();

		if (id == 0)
		{
			notEnd = false;
		}
		else
		{
			_cache->popCubes(id);
		}
	}
}

}
