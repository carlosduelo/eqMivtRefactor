/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <visibleCubes.h>
#include <visibleCubes_cuda.h>

#include <iostream>
#include <algorithm>

namespace eqMivt
{

VisibleCubes::VisibleCubes()
{
	_size = 0;
	_visibleCubes = 0;
	_visibleCubesGPU = 0;

	_cubeC = 0;
	_nocubeC = 0;
	_cachedC = 0;
	_paintedC = 0;
	_cubeG = 0;
	_nocubeG = 0;
	_cachedG = 0;
	_paintedG = 0;
	_indexGPU = 0;
}


VisibleCubes::~VisibleCubes()
{
}

void VisibleCubes::destroy()
{
	if (_visibleCubes != 0)
		if (cudaSuccess != cudaFreeHost((void*)_visibleCubes))
		{
			std::cerr<<"Visible cubes, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}

	if (_visibleCubesGPU != 0)
		if (cudaSuccess != cudaFree((void*)_visibleCubesGPU))
		{
			std::cerr<<"Visible cubes, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}

	if (_cubeC != 0)
		if (cudaSuccess != cudaFreeHost((void*)_cubeC))
		{
			std::cerr<<"Visible cubes, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}
	if (_nocubeC != 0)
		if (cudaSuccess != cudaFreeHost((void*)_nocubeC))
		{
			std::cerr<<"Visible cubes, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}
	if (_cachedC != 0)
		if (cudaSuccess != cudaFreeHost((void*)_cachedC))
		{
			std::cerr<<"Visible cubes, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}
	if (_paintedC != 0)
		if (cudaSuccess != cudaFreeHost((void*)_paintedC))
		{
			std::cerr<<"Visible cubes, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}

	if (_cubeG != 0)
		if (cudaSuccess != cudaFree((void*)_cubeG))
		{
			std::cerr<<"Visible cubes, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}
	if (_nocubeG != 0)
		if (cudaSuccess != cudaFree((void*)_nocubeG))
		{
			std::cerr<<"Visible cubes, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}
	if (_cachedG != 0)
		if (cudaSuccess != cudaFree((void*)_cachedG))
		{
			std::cerr<<"Visible cubes, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}
	if (_paintedG != 0)
		if (cudaSuccess != cudaFree((void*)_paintedG))
		{
			std::cerr<<"Visible cubes, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}
	if (_indexGPU != 0)
		if (cudaSuccess != cudaFree((void*)_indexGPU))
		{
			std::cerr<<"Visible cubes, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}
}

void VisibleCubes::reSize(int numPixels)
{
	_size = numPixels;

	destroy();

	if (cudaSuccess != cudaMalloc((void**)&_visibleCubesGPU, _size*sizeof(visibleCube_t)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	
	if (cudaSuccess != cudaHostAlloc((void**)&_visibleCubes, _size*sizeof(visibleCube_t), cudaHostAllocDefault))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}

	if (cudaSuccess != cudaHostAlloc((void**)&_cubeC, (1 + _size)*sizeof(int), cudaHostAllocDefault))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaHostAlloc((void**)&_nocubeC, (1 + _size)*sizeof(int), cudaHostAllocDefault))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaHostAlloc((void**)&_cachedC, (1 + _size)*sizeof(int), cudaHostAllocDefault))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaHostAlloc((void**)&_paintedC, (1 + _size)*sizeof(int), cudaHostAllocDefault))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}

	if (cudaSuccess != cudaMalloc((void**)&_cubeG, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMalloc((void**)&_nocubeG, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMalloc((void**)&_cachedG, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMalloc((void**)&_paintedG, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMalloc((void**)&_indexGPU, (_size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMemset((void*)_cubeG, 0, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMemset((void*)_nocubeG, 0, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMemset((void*)_cachedG, 0, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMemset((void*)_paintedG, 0, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMemset((void*)_indexGPU, 0, (_size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
}

void VisibleCubes::init()
{
	if (_size == 0)
	{
		std::cerr<<"Visible cubes has to be reSized"<<std::endl;
		throw;
	}

	for(int i=0; i<_size; i++)
	{
		_visibleCubes[i].id = 0;
		_visibleCubes[i].data = 0;
		_visibleCubes[i].state = NOCUBE;
		_visibleCubes[i].cubeID = 0;
		_visibleCubes[i].pixel= i;
		_nocubeC[i+1] = i;
		_cubeC[i+1] = 0;
		_cachedC[i+1] = 0;
		_paintedC[i+1] = 0;
	}
	_nocubeC[0] = _size;
	_cubeC[0] = 0;
	_cachedC[0] = 0;
	_paintedC[0] = 0;

	updateGPU(NOCUBE, 0);
}

void VisibleCubes::updateIndexCPU()
{
	if (cudaSuccess != cudaMemcpy((void*)_visibleCubesGPU, _visibleCubes, _size*sizeof(visibleCube_t), cudaMemcpyHostToDevice))
	{
		std::cerr<<"Visible cubes, error updating cpu copy: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}

	// Init cuda indexs
	if (cudaSuccess != cudaMemset((void*)_cubeG, 0, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMemset((void*)_nocubeG, 0, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMemset((void*)_cachedG, 0, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMemset((void*)_paintedG, 0, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}

	updateIndex(_visibleCubesGPU, _size, _cubeG + 1, _cubeG, _nocubeG + 1, _nocubeG, _cachedG + 1, _cachedG, _paintedG + 1, _paintedG); 

	if (	cudaSuccess != cudaMemcpy((void*)_cubeC, _cubeG, (_size + 1)*sizeof(int), cudaMemcpyDeviceToHost) ||
			cudaSuccess != cudaMemcpy((void*)_nocubeC, _nocubeG, (_size + 1)*sizeof(int), cudaMemcpyDeviceToHost) ||
			cudaSuccess != cudaMemcpy((void*)_cachedC, _cachedG, (_size + 1)*sizeof(int), cudaMemcpyDeviceToHost) ||
			cudaSuccess != cudaMemcpy((void*)_paintedC, _paintedG, (_size + 1)*sizeof(int), cudaMemcpyDeviceToHost) 
			)
	{
		std::cerr<<"Visible cubes, error updating cpu copy: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}

	#ifdef NDEBUG
	for(int i=0; i<_cubeC[0]; i++)
	{
		for(int j=0; j<_nocubeC[0]; j++)
			if (_cubeC[1+i] == _nocubeC[1+j])
			{
				std::cerr<<"Visisble Cubes, updating index not stable"<<std::endl;
				throw;
			}
		for(int j=0; j<_cachedC[0]; j++)
			if (_cubeC[1+i] == _cachedC[1+j])
			{
				std::cerr<<"Visisble Cubes, updating index not stable"<<std::endl;
				throw;
			}
		for(int j=0; j<_paintedC[0]; j++)
			if (_cubeC[1+i] == _paintedC[1+j])
			{
				std::cerr<<"Visisble Cubes, updating index not stable"<<std::endl;
				throw;
			}
	}
	for(int i=0; i<_nocubeC[0]; i++)
	{
		for(int j=0; j<_cubeC[0]; j++)
			if (_nocubeC[1+i] == _cubeC[1+j])
			{
				std::cerr<<"Visisble Cubes, updating index not stable"<<std::endl;
				throw;
			}
		for(int j=0; j<_cachedC[0]; j++)
			if (_nocubeC[1+i] == _cachedC[1+j])
			{
				std::cerr<<"Visisble Cubes, updating index not stable"<<std::endl;
				throw;
			}
		for(int j=0; j<_paintedC[0]; j++)
			if (_nocubeC[1+i] == _paintedC[1+j])
			{
				std::cerr<<"Visisble Cubes, updating index not stable"<<std::endl;
				throw;
			}
	}
	for(int i=0; i<_cachedC[0]; i++)
	{
		for(int j=0; j<_nocubeC[0]; j++)
			if (_cachedC[1+i] == _nocubeC[1+j])
			{
				std::cerr<<"Visisble Cubes, updating index not stable"<<std::endl;
				throw;
			}
		for(int j=0; j<_cubeC[0]; j++)
			if (_cachedC[1+i] == _cubeC[1+j])
			{
				std::cerr<<"Visisble Cubes, updating index not stable"<<std::endl;
				throw;
			}
		for(int j=0; j<_paintedC[0]; j++)
			if (_cachedC[1+i] == _paintedC[1+j])
			{
				std::cerr<<"Visisble Cubes, updating index not stable"<<std::endl;
				throw;
			}
	}
	for(int i=0; i<_paintedC[0]; i++)
	{
		for(int j=0; j<_cubeC[0]; j++)
			if (_paintedC[1+i] == _cubeC[1+j])
			{
				std::cerr<<"Visisble Cubes, updating index not stable"<<std::endl;
				throw;
			}
		for(int j=0; j<_nocubeC[0]; j++)
			if (_paintedC[1+i] == _nocubeC[1+j])
			{
				std::cerr<<"Visisble Cubes, updating index not stable"<<std::endl;
				throw;
			}
		for(int j=0; j<_cachedC[0]; j++)
			if (_paintedC[1+i] == _cachedC[1+j])
			{
				std::cerr<<"Visisble Cubes, updating index not stable"<<std::endl;
				throw;
			}
	}
	#endif
}

void VisibleCubes::updateCPU()
{
	if (cudaSuccess != cudaMemcpy((void*)_visibleCubes, _visibleCubesGPU, _size*sizeof(visibleCube_t), cudaMemcpyDeviceToHost))
	{
		std::cerr<<"Visible cubes, error updating cpu copy: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}

	// Init cuda indexs
	if (cudaSuccess != cudaMemset((void*)_cubeG, 0, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMemset((void*)_nocubeG, 0, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMemset((void*)_cachedG, 0, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMemset((void*)_paintedG, 0, (1 + _size)*sizeof(int)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}

	updateIndex(_visibleCubesGPU, _size, _cubeG + 1, _cubeG, _nocubeG + 1, _nocubeG, _cachedG + 1, _cachedG, _paintedG + 1, _paintedG); 

	if (	cudaSuccess != cudaMemcpy((void*)_cubeC, _cubeG, (_size + 1)*sizeof(int), cudaMemcpyDeviceToHost) ||
			cudaSuccess != cudaMemcpy((void*)_nocubeC, _nocubeG, (_size + 1)*sizeof(int), cudaMemcpyDeviceToHost) ||
			cudaSuccess != cudaMemcpy((void*)_cachedC, _cachedG, (_size + 1)*sizeof(int), cudaMemcpyDeviceToHost) ||
			cudaSuccess != cudaMemcpy((void*)_paintedC, _paintedG, (_size + 1)*sizeof(int), cudaMemcpyDeviceToHost) 
			)
	{
		std::cerr<<"Visible cubes, error updating cpu copy: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
}

void VisibleCubes::updateGPU(statusCube type, cudaStream_t stream)
{
	_sizeGPU = 0;

	if (_cubeC[0] > 0 &&(type & CUBE) != NONE)
	{
		if (cudaSuccess != cudaMemcpyAsync((void*)(_indexGPU + _sizeGPU), _cubeC + 1, _cubeC[0]*sizeof(int), cudaMemcpyHostToDevice))
		{
			std::cerr<<"Visible cubes, error updating cpu copy: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}
		_sizeGPU+= _cubeC[0];
	}
	if (_nocubeC[0] > 0 && (type & NOCUBE) != NONE)
	{
		if (cudaSuccess != cudaMemcpyAsync((void*)(_indexGPU + _sizeGPU), _nocubeC + 1, _nocubeC[0]*sizeof(int), cudaMemcpyHostToDevice))
		{
			std::cerr<<"Visible cubes, error updating cpu copy: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}
		_sizeGPU+=_nocubeC[0];
	}
	if (_cachedC[0] > 0 && (type & CACHED) != NONE)
	{
		if (cudaSuccess != cudaMemcpyAsync((void*)(_indexGPU + _sizeGPU), _cachedC + 1, _cachedC[0]*sizeof(int), cudaMemcpyHostToDevice))
		{
			std::cerr<<"Visible cubes, error updating cpu copy: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}
		_sizeGPU+=_cachedC[0];
	}
	if (_paintedC[0] > 0 && (type & PAINTED) != NONE)
	{
		if (cudaSuccess != cudaMemcpyAsync((void*)(_indexGPU + _sizeGPU), _paintedC + 1, _paintedC[0]*sizeof(int), cudaMemcpyHostToDevice))
		{
			std::cerr<<"Visible cubes, error updating cpu copy: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}
		_sizeGPU+=_paintedC[0];
	}
}

visibleCubeGPU VisibleCubes::getVisibleCubesGPU()
{
	return _visibleCubesGPU; 
}

indexVisibleCubeGPU VisibleCubes::getIndexVisibleCubesGPU()
{
	return _indexGPU; 
}

int VisibleCubes::getSizeGPU()
{
	return _sizeGPU;
}

int VisibleCubes::getSize()
{
	return _size;
}

visibleCube_t * VisibleCubes::getCube(int i)
{
	if (i < _size && i >= 0)
		return &_visibleCubes[i];
	else
	{
		std::cerr<<"Error geting a visible cube, out of range "<<i<<std::endl;
		throw;
	}	
}

std::vector<int> VisibleCubes::getListCubes(statusCube type)
{
	std::vector<int> result;

	if ((type & CUBE) != NONE && _cubeC[0] > 0)
		result.insert(result.end(), &_cubeC[1], &_cubeC[_cubeC[0]+1]);
	if ((type & NOCUBE) != NONE && _nocubeC[0] > 0)
		result.insert(result.end(), &_nocubeC[1], &_nocubeC[_nocubeC[0]+1]);
	if ((type & CACHED) != NONE && _cachedC[0] > 0)
		result.insert(result.end(), &_cachedC[1], &_cachedC[_cachedC[0]+1]);
	if ((type & PAINTED) != NONE && _paintedC[0] > 0)
		result.insert(result.end(), &_paintedC[1], &_paintedC[_paintedC[0]+1]);
	
	return result;
}
		
int VisibleCubes::getNumElements(statusCube type)
{
	int num = 0;

	if ((type & CUBE) != NONE)
		num += _cubeC[0];
	if ((type & NOCUBE) != NONE)
		num += _nocubeC[0];
	if ((type & CACHED) != NONE)
		num += _cachedC[0];
	if ((type & PAINTED) != NONE)
		num +=_paintedC[0];

	return num;
}

}
