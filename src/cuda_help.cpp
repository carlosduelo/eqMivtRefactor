/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <cuda_runtime.h>

namespace eqMivt
{
int getBestDevice()
{
	int num_devices, device;
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 1)
	{
		int max_multiprocessors = 0, max_device = 0;
		for (device = 0; device < num_devices; device++)
		{
			cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, device);
			if (max_multiprocessors < properties.multiProcessorCount)
			{
				max_multiprocessors = properties.multiProcessorCount;
				max_device = device;
			}
		}
		return max_device;
	}
	return 0;
}
}
