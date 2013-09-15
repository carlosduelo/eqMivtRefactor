/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <visibleCubes.h>

#include <iostream>
#include <algorithm>

namespace eqMivt
{

VisibleCubes::VisibleCubes()
{
	_size = 0;
	_visibleCubes = 0;
	_visibleCubesAUX = 0;
	_visibleCubesGPU = 0;

	_cube.clear();
	_nocube.clear();
	_cached.clear();
	_nocached.clear();
	_painted.clear();

}

void VisibleCubes::reSize(int numPixels)
{
	_size = numPixels;
	
	if (_visibleCubes != 0)
		if (cudaSuccess != cudaFreeHost((void*)_visibleCubes))
		{
			std::cerr<<"Visible cubes, error free memory"<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}

	if (_visibleCubesAUX != 0)
		if (cudaSuccess != cudaFreeHost((void*)_visibleCubesAUX))
		{
			std::cerr<<"Visible cubes, error free memory"<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}

	if (_visibleCubesGPU != 0)
		if (cudaSuccess != cudaFreeHost((void*)_visibleCubesGPU))
		{
			std::cerr<<"Visible cubes, error free memory"<<cudaGetErrorString(cudaGetLastError())<<std::endl;	
			throw;
		}

	if (cudaSuccess != cudaMalloc((void**)&_visibleCubesGPU, _size*sizeof(visibleCube_t)))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory"<<std::endl;
		throw;
	}
	
	if (cudaSuccess != cudaHostAlloc((void**)&_visibleCubes, _size*sizeof(visibleCube_t), cudaHostAllocDefault))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory"<<std::endl;
		throw;
	}

	if (cudaSuccess != cudaHostAlloc((void**)&_visibleCubesAUX, _size*sizeof(visibleCube_t), cudaHostAllocDefault))
	{                                                                                               
		std::cerr<<"Visible cubes, error allocating memory"<<std::endl;
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

	_cube.clear();
	_nocube.clear();
	_cached.clear();
	_nocached.clear();
	_painted.clear();

	for(int i=0; i<_size; i++)
	{
		_visibleCubes[i].id = 0;
		_visibleCubes[i].data = 0;
		_visibleCubes[i].state = NOCUBE;
		_visibleCubes[i].cubeID = 0;
		_visibleCubes[i].pixel= i;
		_nocube.push_back(i);
	}

	updateGPU(NOCUBE, false, 0);
}

void VisibleCubes::updateCPU()
{
	if (cudaSuccess != cudaMemcpy((void*)_visibleCubesAUX, _visibleCubesGPU, _sizeGPU*sizeof(visibleCube_t), cudaMemcpyDeviceToHost))
	{
		std::cerr<<"Visible cubes, error updating cpu copy"<<std::endl;
		throw;
	}

	for(int i=0; i<_sizeGPU; i++)
	{
		#ifndef NDEBUG
		if (_visibleCubesAUX[i].pixel >= _size &&
			_visibleCubes[_visibleCubesAUX[i].pixel].pixel != _visibleCubesAUX[i].pixel &&
			_visibleCubes[_visibleCubesAUX[i].pixel].cubeID != _visibleCubesAUX[i].cubeID &&
			_visibleCubes[_visibleCubesAUX[i].pixel].data != _visibleCubesAUX[i].data)
		{
			std::cerr<<"Visible cubes, someting went wrong"<<std::endl;
			throw;
		}
		#endif

		_visibleCubes[_visibleCubesAUX[i].pixel].id = _visibleCubesAUX[i].id;

		if (_visibleCubes[_visibleCubesAUX[i].pixel].state != _visibleCubesAUX[i].state)
		{
			switch(_visibleCubes[_visibleCubesAUX[i].pixel].state)
			{
				case NOCUBE:
					_nocube.erase(std::remove(_nocube.begin(), _nocube.end(), _visibleCubesAUX[i].pixel), _nocube.end());
					break;
				case CUBE:
					_cube.erase(std::remove(_cube.begin(), _cube.end(), _visibleCubesAUX[i].pixel), _cube.end());
					break;
				case CACHED:
					_cached.erase(std::remove(_cached.begin(), _cached.end(), _visibleCubesAUX[i].pixel), _cached.end());
					break;
				case NOCACHED:
					_nocached.erase(std::remove(_nocached.begin(), _nocached.end(), _visibleCubesAUX[i].pixel), _nocached.end());
					break;
				case PAINTED:
					_painted.erase(std::remove(_painted.begin(), _painted.end(), _visibleCubesAUX[i].pixel), _painted.end());
					break;
				#ifndef NDEBUG
				default:
					std::cerr<<"Visible cubes, error updateing cpu"<<std::endl;
					throw;
				#endif
			}

			switch (_visibleCubesAUX[i].state)
			{
				case NOCUBE:
					_nocube.push_back(_visibleCubesAUX[i].pixel);
					break;
				case CUBE:
					_cube.push_back(_visibleCubesAUX[i].pixel);
					break;
				case CACHED:
					_cached.push_back(_visibleCubesAUX[i].pixel);
					break;
				case NOCACHED:
					_nocached.push_back(_visibleCubesAUX[i].pixel);
					break;
				case PAINTED:
					_painted.push_back(_visibleCubesAUX[i].pixel);
					break;
				#ifndef NDEBUG
				default:
					std::cerr<<"Visible cubes, error updateing cpu"<<std::endl;
					throw;
				#endif
			}

			_visibleCubes[_visibleCubesAUX[i].pixel].state = _visibleCubesAUX[i].state;
		}
	}

	#ifndef NDEBUG
	int cubes = _cube.size();
	int nocubes = _nocube.size();
	int cached = _cached.size();
	int nocached = _nocached.size();
	int painted = _painted.size();
	for(int i=0; i<_size; i++)
	{
		switch (_visibleCubes[i].state)
		{
			case NOCUBE:
				nocubes--;
				break;
			case CUBE:
				cubes--;
				break;
			case CACHED:
				cached--;
				break;
			case NOCACHED:
				nocached--;
				break;
			case PAINTED:
				painted--;
				break;
			#ifndef NDEBUG
			default:
				std::cerr<<"Visible cubes, error updating cpu"<<std::endl;
				throw;
			#endif
		}
	}
	if (cubes != 0 || nocubes!=0 || cached != 0 || nocached != 0 || painted != 0)
	{
		std::cerr<<"Visible cubes, error updating CPU "<<cubes<<" "<<nocubes<<" "<<cached<<" "<<nocached<<" "<<painted<<std::endl;
		throw;
	}
	#endif
}

void VisibleCubes::updateGPU(unsigned char type, bool sync, cudaStream_t stream)
{
	_sizeGPU = 0;

	if ((type & CUBE) != NONE)
		for(int i=0; i<_cube.size(); i++)
		{
			_visibleCubesAUX[_sizeGPU] = _visibleCubes[_cube[i]];
			_sizeGPU++;
		}
	if ((type & NOCUBE) != NONE)
		for(int i=0; i<_nocube.size(); i++)
		{
			_visibleCubesAUX[_sizeGPU] = _visibleCubes[_nocube[i]];
			_sizeGPU++;
		}
	if ((type & CACHED) != NONE)
		for(int i=0; i<_cached.size(); i++)
		{
			_visibleCubesAUX[_sizeGPU] = _visibleCubes[_cached[i]];
			_sizeGPU++;
		}
	if ((type & NOCACHED) != NONE)
		for(int i=0; i<_nocached.size(); i++)
		{
			_visibleCubesAUX[_sizeGPU] = _visibleCubes[_nocached[i]];
			_sizeGPU++;
		}
	if ((type & PAINTED) != NONE)
		for(int i=0; i<_painted.size(); i++)
		{
			_visibleCubesAUX[_sizeGPU] = _visibleCubes[_painted[i]];
			_sizeGPU++;
		}

	if (sync)
	{
		if (cudaSuccess != cudaMemcpy((void*)_visibleCubesGPU, _visibleCubesAUX, _sizeGPU*sizeof(visibleCube_t), cudaMemcpyHostToDevice))
		{
			std::cerr<<"Visible cubes, error updating cpu copy"<<std::endl;
			throw;
		}
	}
	else
	{
			if (cudaSuccess != cudaMemcpyAsync((void*)_visibleCubesGPU, _visibleCubesAUX, _sizeGPU*sizeof(visibleCube_t), cudaMemcpyHostToDevice, stream))
			{
				std::cerr<<"Visible cubes, error updating cpu copy"<<std::endl;
				throw;
			}
	}
}

visibleCubeGPU VisibleCubes::getVisibleCubesGPU()
{
	return _visibleCubesGPU; 
}

int VisibleCubes::getSizeGPU()
{
	return _sizeGPU;
}

void VisibleCubes::updateCube(int iter, int idCube, int state)
{
	_visibleCubes[iter].cubeID	= idCube;
	switch(_visibleCubes[iter].state)                                                                                 
	{                                                                                                                 
		case NOCUBE:
			_nocube.erase(std::remove(_nocube.begin(), _nocube.end(), iter), _nocube.end());                           
			break;
		case CUBE:                                                                                                    
			_cube.erase(std::remove(_nocube.begin(), _nocube.end(), iter), _nocube.end());                             
			break;                                                                                                    
		case CACHED:
			_cached.erase(std::remove(_nocube.begin(), _nocube.end(), iter), _nocube.end());                           
			break;
		case NOCACHED:                                                                                                
			_nocached.erase(std::remove(_nocube.begin(), _nocube.end(), iter), _nocube.end());                         
			break;                                                                                                    
		case PAINTED:
			_painted.erase(std::remove(_nocube.begin(), _nocube.end(), iter), _nocube.end());                          
			break;                                                                                                    
		#ifndef NDEBUG                                                                                                
		default:                
			std::cerr<<"Visible cubes, error updateing cpu"<<std::endl;                                               
			throw;
		#endif                                                                                                        
	}       

	switch (state)
	{       
		case NOCUBE:
			_nocube.push_back(iter);                                                                                  
			break;              
		case CUBE:
			_cube.push_back(iter);                                                                                    
			break;                                                                                                    
		case CACHED:                                                                                                  
			_cached.push_back(iter);                                                                                  
			break;
		case NOCACHED:
			_nocached.push_back(iter);
			break;
		case PAINTED:
			_painted.push_back(iter);
			break;
		#ifndef NDEBUG
		default:
			std::cerr<<"Visible cubes, error updateing cpu"<<std::endl;
			throw;
		#endif
	}

	_visibleCubes[iter].state	= state;
}

visibleCube_t VisibleCubes::getCube(int i)
{
	if (i < _size && i > 0)
		return _visibleCubes[i];
	else
	{
		std::cerr<<"Error geting a visible cube, out of range"<<std::endl;
		throw;
	}	
}

std::vector<int> VisibleCubes::getListCubes(unsigned char type)
{
	std::vector<int> result;

	if ((type & CUBE) != NONE)
		result.insert(result.end(), _cube.begin(), _cube.end());
	if ((type & NOCUBE) != NONE)
		result.insert(result.end(), _nocube.begin(), _nocube.end());
	if ((type & CACHED) != NONE)
		result.insert(result.end(), _cached.begin(), _cached.end());
	if ((type & NOCACHED) != NONE)
		result.insert(result.end(), _nocached.begin(), _nocached.end());
	if ((type & PAINTED) != NONE)
		result.insert(result.end(), _painted.begin(), _painted.end());
	
	return result;
}

void	VisibleCubes::updateVisibleCubes(std::vector<updateCube_t> list)
{

	for(std::vector< updateCube_t >::iterator it=list.begin(); it!=list.end(); ++it)
	{
		updateCube(it->pixel, it->cubeID, it->state);
	}

}

}
