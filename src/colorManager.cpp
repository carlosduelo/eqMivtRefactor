/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <colorManager.h>

#include <iostream>
#include <fstream>

#include <cuda_runtime.h>

namespace eqMivt
{

bool ColorManager::init(std::string file_name)
{
	_colorsC = new float[3*NUM_COLORS+3];

	if (file_name == "")
	{
		for(int p=0; p<NUM_COLORS; p++)
			_colorsC[p] = 1.0f;

		for(int p=0; p<64; p++)
			_colorsC[(NUM_COLORS+1) + p] = 0.0f;

		float dc = 1.0f/((float)NUM_COLORS - 60.0f);
		int k = 1;
		for(int p=64; p<NUM_COLORS; p++)
		{
			_colorsC[(NUM_COLORS+1) + p] = (float)k*dc; 
			k++;
		}

		for(int p=0; p<192; p++)
			_colorsC[2*(NUM_COLORS+1) + p] = 0.0f;

		dc = 1.0f/100.0f;
		k=1;
		for(int p=192; p<NUM_COLORS; p++)
		{
			_colorsC[(2*(NUM_COLORS+1)) + p] = (float)k*dc; 
			k++;
		}
		_colorsC[NUM_COLORS] = 1.0f;
		_colorsC[2*NUM_COLORS+1] = 1.0f;
		_colorsC[3*NUM_COLORS+2] = 1.0f;
	}
	else
	{
		std::ifstream file;
		try
		{
			file.open(file_name.c_str(), std::ifstream::binary);
			file.read((char*)_colorsC, (3*NUM_COLORS+3)*sizeof(float));
			file.close();
		}
		catch(...)
		{
			std::cerr<<"Error opening transfer function color file"<<std::endl;
			return false;
		}
	}

	return true;
}

void ColorManager::destroy()
{
	delete[] _colorsC;

	boost::unordered_map<device_t, color_t>::iterator it = _colors.begin();
	while(it != _colors.end())
	{
		if (cudaSuccess != cudaSetDevice(it->first))
		{
			std::cerr<<"Color Manager, cudaSetDevice error: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return;
		}
		if (cudaSuccess != cudaFree(it->second.r))
		{
			std::cerr<<"Color Manager, cudaFree error: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			return;
		}

		it++;
	}
}

color_t ColorManager::getColors(device_t device)
{
	boost::unordered_map<device_t, color_t>::iterator it = _colors.find(device);
	if (it != _colors.end())
		return it->second;

	color_t c;
	
	if (cudaSuccess != cudaSetDevice(device))
	{
		std::cerr<<"Color Manager, cudaSetDevice error: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMalloc((void**)&c.r, (3*NUM_COLORS + 3)*sizeof(float)))
	{
		std::cerr<<"Error creating colors"<<std::endl;
		throw;
	}
	if (cudaSuccess != cudaMemcpy((void*)c.r, (void*)_colorsC, (3*NUM_COLORS + 3)*sizeof(float), cudaMemcpyHostToDevice))
	{
		std::cerr<<"Error creating colors"<<std::endl;
		throw;
	}

	c.g = c.r + NUM_COLORS + 1;
	c.b = c.r + 2*(NUM_COLORS + 1);

	_colors.insert(std::make_pair<device_t, color_t>(device, c));

	return c;
}
}
