/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <renderOGL.h>

namespace eqMivt
{

bool RenderOGL::init(device_t device, std::string name)
{
	_name = name;
	_frame = 0;
	_cuda_pbo_resource = 0;

	return Render::init(device);
}

void RenderOGL::destroy()
{
    if (_cuda_pbo_resource != 0 && cudaSuccess != cudaGraphicsUnregisterResource(_cuda_pbo_resource))
    {
    	std::cerr<<"Error cudaGraphicsUnregisterResource"<<std::endl;
    }

	Render::destroy();
}

bool RenderOGL::setViewPort(int pvpW, int pvpH, GLuint pbo)
{
    // Resize pbo
    if (_cuda_pbo_resource != 0 && cudaSuccess != cudaGraphicsUnregisterResource(_cuda_pbo_resource))
    {
    	std::cerr<<"Error cudaGraphicsUnregisterResource"<<std::endl;
		return false;
    }
    if (cudaSuccess != cudaGraphicsGLRegisterBuffer(&_cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard))
    {
    	std::cerr<<"Error cudaGraphicsGLRegisterBuffer"<<std::endl;
		return false;
    }

	return Render::setViewPort(pvpW, pvpH);
}
bool RenderOGL::frameDraw(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
							vmml::vector<4, float> up, vmml::vector<4, float> right,
							float w, float h)
{
    if (cudaSuccess != cudaGraphicsMapResources(1, &_cuda_pbo_resource, 0))
    {
    	std::cerr<<"Error cudaGraphicsMapResources"<<std::endl;
		return false;
    }

    size_t num_bytes;
    if (cudaSuccess != cudaGraphicsResourceGetMappedPointer((void **)&_pixelBuffer, &num_bytes, _cuda_pbo_resource))
    {
    	std::cerr<<"Error cudaGraphicsResourceGetMappedPointer"<<std::endl;
		return false;
    }

	if (cudaSuccess != cudaMemset((void*)_pixelBuffer, 0, 3*_pvpW*_pvpH*sizeof(float)))
	{
		std::cerr<<"RenderPNG, error init pixel buffer "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}

	Render::frameDraw(origin, LB, up, right, w, h);

    if (cudaSuccess != cudaGraphicsUnmapResources(1, &_cuda_pbo_resource, 0))
    {
    	std::cerr<<"Error cudaGraphicsUnmapResources"<<std::endl;
    }

	_frame++;

	return true;
}
}
