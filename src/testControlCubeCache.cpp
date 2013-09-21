/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlCubeCache.h>

#include <mortonCodeUtil_CPU.h>

#include <cuda_runtime.h>

#include <lunchbox/sleep.h>

#include <boost/progress.hpp>


eqMivt::ControlPlaneCache cpc;
eqMivt::hdf5File hdf5File;
eqMivt::ControlCubeCache ccc;

bool test(int nLevels, int levelCube, vmml::vector<3,int> offset)
{
	ccc.reSize(nLevels, levelCube, offset);

	int dimC = exp2(nLevels-levelCube) + 2 * CUBE_INC;
	int dimV = exp2(nLevels);

	float * cube = new float[dimC*dimC*dimC];
	float * cubeC = new float[dimC*dimC*dimC];
	float * cubeG = 0;

	eqMivt::index_node_t idS = eqMivt::coordinateToIndex(vmml::vector<3,int>(0,0,0), levelCube, nLevels);
	eqMivt::index_node_t idF = eqMivt::coordinateToIndex(vmml::vector<3,int>(dimV-1, dimV-1, dimV-1), levelCube, nLevels);

	bool error = false;
	#ifdef NDEBUG
	#ifndef DISK_TIMING 
		boost::progress_display show_progress(idF - idS + 1);
	#endif
	#endif

	for(eqMivt::index_node_t id=idS; id<=idF && !error; id++)
	{
		vmml::vector<3,int> coord = eqMivt::getMinBoxIndex2(id, levelCube, nLevels) - vmml::vector<3,int>(CUBE_INC, CUBE_INC, CUBE_INC);
		do
		{
			cubeG = ccc.getAndBlockCube(id);
			lunchbox::sleep(50);	
		}
		while(cubeG == 0);

		#ifndef NDEBUG
		std::cout<<id<< " dir "<<cubeG<<std::endl;
		#endif

		if (cudaSuccess != cudaMemcpy((void*)cube, (void*)cubeG, dimC*dimC*dimC*sizeof(float), cudaMemcpyDeviceToHost))
		{
			std::cerr<<"Error copying cube to CPU: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}

		hdf5File.readCube(id, cubeC, levelCube, nLevels, vmml::vector<3,int>(dimC,dimC,dimC), offset); 

		for(int i= 0; i<dimC; i++)
			for(int j=0; j<dimC; j++)
				for(int k=0; k<dimC; k++)
				{
					if (cube[i*dimC*dimC+j*dimC+k] != cubeC[i*dimC*dimC+j*dimC+k])
					{
						std::cerr<<"Not coincidence("<<coord.x() + i<<","<<coord.y() + j<<","<<coord.z() + k<<") "<<cube[i*dimC*dimC+j*dimC+k]<<" "<<cubeC[i*dimC*dimC+j*dimC+k]<<std::endl;
						error = true;
					}
				}

		ccc.unlockCube(id);
		
		if (error)
		{
			vmml::vector<3, int> cs = eqMivt::getMinBoxIndex2(id,levelCube, nLevels) - vmml::vector<3, int>(CUBE_INC, CUBE_INC, CUBE_INC);
			vmml::vector<3, int> ce = cs + vmml::vector<3, int>(dimC,dimC, dimC);
			std::cerr<<"Cube id "<<id<<" coordinates "<<cs<<" "<<ce<<" nLevels "<<nLevels<<" levelCube "<<levelCube<<" offset "<<offset<<std::endl;
		}
		#ifdef NDEBUG
		#ifndef DISK_TIMING 
			++show_progress;
		#endif
		#endif
	}


	if (!error)
		std::cout<<"Test OK"<<std::endl;
	else
		std::cout<<"Test Fail!"<<std::endl;

	delete[] cube;
	delete[] cubeC;

	return error;
}



int main(int argc, char ** argv)
{
	int nLevels = 10;
	int levelCube = 8;
	vmml::vector<3,int> offset(0,0,0);


	std::vector<std::string> parameters;
	parameters.push_back(std::string(argv[1]));
	parameters.push_back(std::string(argv[2]));

	cpc.initParameter(parameters);
	hdf5File.init(parameters);
	ccc.initParameter(&cpc);

	cpc.start();
	ccc.start();
	lunchbox::sleep(500);

	bool error =	!test(2, 2, vmml::vector<3,int>(0,0,0)) && 
					!test(10, 1, vmml::vector<3,int>(0,0,0)) &&
					!test(10, 2, vmml::vector<3,int>(0,0,0)) &&
					!test(10, 3, vmml::vector<3,int>(0,0,0)) &&
					!test(10, 4, vmml::vector<3,int>(0,0,0)) &&
					!test(10, 5, vmml::vector<3,int>(0,0,0)) &&
					!test(10, 6, vmml::vector<3,int>(0,0,0)) &&
					!test(10, 10, vmml::vector<3,int>(0,0,0)) &&
					!test(2, 2, vmml::vector<3,int>(255,0,123)) &&
					!test(10, 1, vmml::vector<3,int>(123,0,0)) &&
					!test(10, 2, vmml::vector<3,int>(321,0,2)) &&
					!test(10, 3, vmml::vector<3,int>(12,0,300)) &&
					!test(10, 4, vmml::vector<3,int>(42,0,99)) &&
					!test(10, 5, vmml::vector<3,int>(12,0,100)) &&
					!test(10, 6, vmml::vector<3,int>(50,0,30)) &&
					!test(10, 10, vmml::vector<3,int>(60,0,0)) &&
					!test(10, 10, vmml::vector<3,int>(90,0,12));


	ccc.stopProcessing();
	cpc.stopProcessing();

	lunchbox::sleep(5000);

	if (error)
		std::cout<<"Test Fail"<<std::endl;
	else
		std::cout<<"Test OK"<<std::endl;

	return 0;
}
