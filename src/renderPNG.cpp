/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <renderPNG.h>

#include <FreeImage.h>

namespace eqMivt
{

bool RenderPNG::init(device_t device)
{
	FreeImage_Initialise();

	_frame = 0;
	_bufferC = 0;
	return Render::init(device);
}

bool RenderPNG::setViewPort(int pvpW, int pvpH)
{
	if (Render::setViewPort(pvpW, pvpH))
	{
		if (_bufferC != 0)
			delete[] _bufferC;
		if (_pixelBuffer != 0)
			if (cudaSuccess != cudaFree((void*)_pixelBuffer))
			{
				std::cerr<<"RenderPNG, error resizing viewport"<<cudaGetErrorString(cudaGetLastError())<<std::endl;
				return false;
			}

		_bufferC = new float[3*_pvpW*_pvpH];
		if (cudaSuccess != cudaMalloc((void**)&_pixelBuffer, 3*_pvpW*_pvpH*sizeof(float)))
		{
			std::cerr<<"RenderPNG, error resizing viewport"<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return false;
		}
		return true;
	}
	else
		return false;
}

void RenderPNG::destroy()
{
	FreeImage_DeInitialise();

	if (_bufferC != 0)
		delete[] _bufferC;
	if (_pixelBuffer != 0)
		if (cudaSuccess != cudaFree((void*)_pixelBuffer))
		{
			std::cerr<<"RenderPNG, error destroying"<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}

	Render::destroy();
}

bool RenderPNG::frameDraw(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
							vmml::vector<4, float> up, vmml::vector<4, float> right,
							float w, float h)
{

	if (cudaSuccess != cudaMemset((void*)_pixelBuffer, 0, 3*_pvpW*_pvpH*sizeof(float)))
	{
		std::cerr<<"RenderPNG, error init pixel buffer "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		return false;
	}

	Render::frameDraw(origin, LB, up, right, w, h);

	if (cudaSuccess != cudaMemcpy((void*)_bufferC, _pixelBuffer, 3*_pvpW*_pvpH*sizeof(float), cudaMemcpyDeviceToHost))
	{
		std::cerr<<"RenderPNG, error copying pixel buffer "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
	}

	/* CREATE PNG */
	FIBITMAP * bitmap = FreeImage_Allocate(_pvpW, _pvpH, 24);
	RGBQUAD color;
	for(int i=0; i<_pvpH; i++)
	{
		for(int j=0; j<_pvpH; j++)
		{
			int id = i*_pvpW+ j;
			color.rgbRed	= _bufferC[id*3]*255;
			color.rgbGreen	= _bufferC[id*3+1]*255;
			color.rgbBlue	= _bufferC[id*3+2]*255;
			FreeImage_SetPixelColor(bitmap, j, i, &color);
		}
	}
	std::stringstream name;
	name<<"frame"<<_frame<<".png";
	FreeImage_Save(FIF_PNG, bitmap, name.str().c_str(), 0);

	delete bitmap;
	/* END CREATE PNG */

	_frame++;

	return true;
}

}
