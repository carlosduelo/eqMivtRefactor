/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlCubeCache.h>

#include <mortonCodeUtil_CPU.h>

#include <cuda_runtime.h>

namespace eqMivt
{

bool ControlCubeCache::initParameter(ControlCubeCPUCache * cpuCache, device_t device)
{
	_cpuCache = cpuCache;
	_device = device;

	return  ControlElementCache::_init();
}

bool ControlCubeCache::_threadInit()
{
	_offset.set(0,0,0);
	_nextOffset.set(-1,-1,-1);
	_nextnLevels = -1;
	_nextLevelCube = -1;
	_nLevels = 0;
	_levelCube = 0;
	_dimCube = 0;
	_maxNumCubes = 0;

	if (cudaSuccess != cudaSetDevice(_device))
	{
		std::cerr<<"Control Cube Cache, error setting device: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}

	return ControlElementCache::_threadInit();
}

void ControlCubeCache::_threadStop()
{
	_freeCache();
	ControlElementCache::_threadStop();
}

void ControlCubeCache::_freeCache()
{
	if (cudaSuccess != cudaSetDevice(_device))
	{
		std::cerr<<"Control Cube Cache, error setting device: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (_memory != 0)
	{
		if(cudaSuccess != cudaFree((void*)_memory))
		{
			std::cerr<<"Control Cubes Cache, error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}
		_memory = 0;
	}

	#ifdef TIMING
	int dim = exp2(_nLevels);
	index_node_t s = coordinateToIndex(vmml::vector<3, int>(0,0,0), _levelCube, _nLevels);
	index_node_t e = coordinateToIndex(vmml::vector<3, int>(dim-1, dim-1, dim-1), _levelCube, _nLevels);
	std::cout<<"==== CONTROL CUBE CACHE ====="<<std::endl;
	std::cout<<"Max num cubes "<< _maxNumCubes<<" cubes form "<<s<<" to "<<e<<" total "<<e-s+1<<std::endl;
	#endif

	ControlElementCache::_freeCache();

	#ifdef TIMING
	std::cout<<"=============================="<<std::endl;
	#endif

}

void ControlCubeCache::_reSizeCache()
{
	_nLevels = _nextnLevels;
	_levelCube = _nextLevelCube;
	_offset	= _nextOffset;
	_nextnLevels = 0;
	_nextLevelCube = 0;

	_dimCube = exp2(_nLevels - _levelCube) + 2 * CUBE_INC;

	_sizeElement = pow(_dimCube, 3); 

	if (cudaSuccess != cudaSetDevice(_device))
	{
		std::cerr<<"Control Cube Cache, error setting device: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (_memory != 0)
		if (cudaSuccess != cudaFree((void*)_memory))
		{                                                                                               
			std::cerr<<"Control Cube Cache, error resizing cache: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}
	size_t total = 0;
	size_t free = 0;

	if (cudaSuccess != cudaMemGetInfo(&free, &total))
	{
		std::cerr<<"Control Cube Cache, error resizing cache: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}

	float memorySize = (0.80f*free); // Get 80% of free memory

	_maxNumCubes = memorySize/ (_sizeElement*sizeof(float));
	if (_maxNumCubes == 0)
	{
		std::cerr<<"Control Cube Cache: Memory aviable is not enough "<<memorySize/1024/1024<<" MB"<<std::endl;
		throw;
	}

	if (cudaSuccess != cudaMalloc((void**)&_memory, _maxNumCubes*_sizeElement*sizeof(float)))
	{
		std::cerr<<"Control Cube Cache, error resizing cache: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}

	_freeSlots = _maxNumCubes;

	ControlElementCache::_reSizeCache();
}

bool ControlCubeCache::_readElement(NodeLinkedList<index_node_t> * element)
{
	index_node_t idCube = element->id;
	index_node_t idCubeCPU = idCube >> 3*(_levelCube - _cpuCache->getCubeLevel()); 

	float * pCube = _cpuCache->getAndBlockElement(idCubeCPU);
	float * cube = (_memory + element->element*_sizeElement);

	if (pCube != 0)
	{
		vmml::vector<3, int> coord = getMinBoxIndex2(idCube, _levelCube, _nLevels);
		vmml::vector<3, int> coordC = getMinBoxIndex2(idCubeCPU, _cpuCache->getCubeLevel(), _nLevels);
		coord -= coordC;
		vmml::vector<3, int> realDimCPU = _cpuCache->getRealCubeDim();

		cudaMemcpy3DParms myParms = {0};
		myParms.srcPtr = make_cudaPitchedPtr((void*)pCube, realDimCPU.z()*sizeof(float), realDimCPU.x(), realDimCPU.y()); 
		//myParms.dstPtr = make_cudaPitchedPtr((void*)cube, _realcubeDim.z()*sizeof(float), _realcubeDim.x(), _realcubeDim.y()); 
		myParms.dstPtr = make_cudaPitchedPtr((void*)cube, _dimCube*sizeof(float), _dimCube, _dimCube); 
		myParms.extent = make_cudaExtent(_dimCube*sizeof(float), _dimCube, _dimCube);
		myParms.dstPos = make_cudaPos(0,0,0);
		myParms.srcPos = make_cudaPos(coord.z()*sizeof(float), coord.y(), coord.x());
		myParms.kind = cudaMemcpyHostToDevice;

		if (cudaSuccess != cudaMemcpy3D(&myParms))
		{
			LBERROR<<"Control Cube Cache: error copying to a device: "<<cudaGetErrorString(cudaGetLastError()) <<" "<<cube<<" "<<pCube<<" "<<_sizeElement<<std::endl;
			return false;
		}

		return true;
	}
	else
	{
		return false;
	}
}

bool ControlCubeCache::freeCacheAndPause()
{
	return ControlElementCache::_freeCacheAndPause();
}

bool ControlCubeCache::reSizeCacheAndContinue(int nLevels, int levelCube, vmml::vector<3, int> offset)
{
	_nextOffset	= offset;
	_nextnLevels = nLevels;
	_nextLevelCube = levelCube;

	return ControlElementCache::_reSizeCacheAndContinue();
}

}

