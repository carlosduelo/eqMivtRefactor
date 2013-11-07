/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <textures.h>

#include <cuda.h>

#include <iostream>

#ifndef EQ_MIVT_TEXTURE_H
#define EQ_MIVT_TEXTURE_H

namespace eqMivt
{
	cudaChannelFormatDesc XchannelDesc;
	cudaChannelFormatDesc YchannelDesc;
	cudaChannelFormatDesc ZchannelDesc;

	texture<float> xgrid;
	texture<float> ygrid;
	texture<float> zgrid;

	bool initTextures()
	{
		XchannelDesc = cudaCreateChannelDesc<float>();
		YchannelDesc = cudaCreateChannelDesc<float>();
		ZchannelDesc = cudaCreateChannelDesc<float>();
		return true;
	}

	bool bindTextures(float * xGrid, float * yGrid, float * zGrid, int3 realDim)
	{
		if (cudaSuccess != cudaBindTexture(NULL, &xgrid, (void*)xGrid, &XchannelDesc, (2*CUBE_INC + realDim.x)*sizeof(float)) ||
			cudaSuccess != cudaBindTexture(NULL, &ygrid, (void*)yGrid, &YchannelDesc, (2*CUBE_INC + realDim.y)*sizeof(float)) ||
			cudaSuccess != cudaBindTexture(NULL, &zgrid, (void*)zGrid, &ZchannelDesc, (2*CUBE_INC + realDim.z)*sizeof(float)))
			{
				std::cerr<<"Error binding texture memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
				return false;
			}

		return true;
	}

	bool unBindTextures()
	{
		if (cudaSuccess != cudaUnbindTexture(xgrid) || 
			cudaSuccess != cudaUnbindTexture(ygrid) || 
			cudaSuccess != cudaUnbindTexture(zgrid))
		{
			std::cerr<<"Error unbinding texture memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return false;
		}
		return true;
	}
}
#endif /* EQ_MIVT_TEXTURE_H*/
