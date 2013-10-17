/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlCubeCache.h>

#include <mortonCodeUtil_CPU.h>

#include <cuda_help.h>

#include <lunchbox/sleep.h>
#include <lunchbox/clock.h>

#include <boost/progress.hpp>


eqMivt::ControlPlaneCache cpc;
eqMivt::hdf5File hdf5File;
eqMivt::ControlCubeCache ccc;

bool test(int nLevels, int levelCube, vmml::vector<3,int> offset)
{
	int dim = exp2(nLevels - levelCube); 
	int dimC = dim + 2 * CUBE_INC;
	int dimV = exp2(nLevels);

	float * cube = new float[dimC*dimC*dimC];
	float * cubeC = new float[dimC*dimC*dimC];
	float * cubeG = 0;
	vmml::vector<3,int> sP;
	vmml::vector<3,int> eP;
	vmml::vector<3,int> mD = hdf5File.getRealDimension();

	sP[0] = offset.x() - CUBE_INC < 0 ? 0 : offset.x() - CUBE_INC; 
	sP[1] = offset.y() - CUBE_INC < 0 ? 0 : offset.y() - CUBE_INC; 
	sP[2] = offset.z() - CUBE_INC < 0 ? 0 : offset.z() - CUBE_INC; 
	eP[0] = offset.x() + dimV + CUBE_INC >= mD.x() ? mD.x() : offset.x() + dimV + CUBE_INC;
	eP[1] = offset.y() + dimV + CUBE_INC >= mD.y() ? mD.y() : offset.y() + dimV + CUBE_INC;
	eP[2] = offset.z() + dimV + CUBE_INC >= mD.z() ? mD.z() : offset.z() + dimV + CUBE_INC;

	std::cout<<"ReSize Plane Cache "<<sP<<" "<<eP<<std::endl;
	std::cout<<"Subset volume "<<offset - vmml::vector<3,int>(CUBE_INC,CUBE_INC,CUBE_INC)<<" "<<offset+vmml::vector<3,int>(dimV+CUBE_INC, dimV+CUBE_INC,dimV+CUBE_INC)<<std::endl;
	std::cout<<"ReSize Cube Cache nLevels "<<nLevels<<" level cube "<<levelCube<<" offset "<<offset<<std::endl;

	if (!cpc.freeCacheAndPause() || !cpc.reSizeCacheAndContinue(sP, eP))
	{
		std::cerr<<"Error, resizing plane cache"<<std::endl;
		return true;
	}
	if (!ccc.freeCacheAndPause() || !ccc.reSizeCacheAndContinue(nLevels, levelCube, offset))
	{
		std::cerr<<"Error, resizing plane cache"<<std::endl;
		return true;
	}

	eqMivt::index_node_t idS = eqMivt::coordinateToIndex(vmml::vector<3,int>(0,0,0), levelCube, nLevels);
	eqMivt::index_node_t idF = eqMivt::coordinateToIndex(vmml::vector<3,int>(dimV-1, dimV-1, dimV-1), levelCube, nLevels);

	bool error = false;
	#ifndef DISK_TIMING 
		boost::progress_display show_progress(idF - idS + 1);
	#endif

	for(eqMivt::index_node_t id=idS; id<=idF && !error; id++)
	{
		vmml::vector<3,int> coord = eqMivt::getMinBoxIndex2(id, levelCube, nLevels) + offset - vmml::vector<3,int>(CUBE_INC, CUBE_INC, CUBE_INC);
		do
		{
			cubeG = ccc.getAndBlockCube(id);
			lunchbox::sleep(50);	
		}
		while(cubeG == 0);

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
			vmml::vector<3, int> cs = eqMivt::getMinBoxIndex2(id,levelCube, nLevels) + offset - vmml::vector<3, int>(CUBE_INC, CUBE_INC, CUBE_INC);
			vmml::vector<3, int> ce = cs + vmml::vector<3, int>(dimC,dimC, dimC);
			std::cerr<<"Cube id "<<id<<" coordinates "<<cs<<" "<<ce<<" nLevels "<<nLevels<<" levelCube "<<levelCube<<" offset "<<offset<<std::endl;
		}
		#ifndef DISK_TIMING 
			++show_progress;
		#endif
	}

	delete[] cube;
	delete[] cubeC;

	return error;
}


void testPerf(int nLevels, int levelCube, vmml::vector<3,int> offset)
{
	int dimV = exp2(nLevels);

	float * cubeG = 0;
	vmml::vector<3,int> sP;
	vmml::vector<3,int> eP;
	vmml::vector<3,int> mD = hdf5File.getRealDimension();

	sP[0] = offset.x() - CUBE_INC < 0 ? 0 : offset.x() - CUBE_INC; 
	sP[1] = offset.y() - CUBE_INC < 0 ? 0 : offset.y() - CUBE_INC; 
	sP[2] = offset.z() - CUBE_INC < 0 ? 0 : offset.z() - CUBE_INC; 
	eP[0] = offset.x() + dimV + CUBE_INC >= mD.x() ? mD.x() : offset.x() + dimV + CUBE_INC;
	eP[1] = offset.y() + dimV + CUBE_INC >= mD.y() ? mD.y() : offset.y() + dimV + CUBE_INC;
	eP[2] = offset.z() + dimV + CUBE_INC >= mD.z() ? mD.z() : offset.z() + dimV + CUBE_INC;

	std::cout<<"ReSize Plane Cache "<<sP<<" "<<eP<<std::endl;
	std::cout<<"Subset volume "<<offset - vmml::vector<3,int>(CUBE_INC,CUBE_INC,CUBE_INC)<<" "<<offset+vmml::vector<3,int>(dimV+CUBE_INC, dimV+CUBE_INC,dimV+CUBE_INC)<<std::endl;
	std::cout<<"ReSize Cube Cache nLevels "<<nLevels<<" level cube "<<levelCube<<" offset "<<offset<<std::endl;

	if (!cpc.freeCacheAndPause() || !cpc.reSizeCacheAndContinue(sP, eP))
	{
		std::cerr<<"Error, resizing plane cache"<<std::endl;
		return;
	}
	if (!ccc.freeCacheAndPause() || !ccc.reSizeCacheAndContinue(nLevels, levelCube, offset))
	{
		std::cerr<<"Error, resizing plane cache"<<std::endl;
		return;
	}

	eqMivt::index_node_t idS = eqMivt::coordinateToIndex(vmml::vector<3,int>(0,0,0), levelCube, nLevels);
	eqMivt::index_node_t idF = eqMivt::coordinateToIndex(vmml::vector<3,int>(dimV-1, dimV-1, dimV-1), levelCube, nLevels);

	#ifndef DISK_TIMING 
		boost::progress_display show_progress(idF - idS + 1);
	#endif

	for(eqMivt::index_node_t id=idS; id<=idF; id++)
	{
		vmml::vector<3,int> coord = eqMivt::getMinBoxIndex2(id, levelCube, nLevels) + offset - vmml::vector<3,int>(CUBE_INC, CUBE_INC, CUBE_INC);
		if (coord.x() < mD.x() && coord.y() < mD.y() && coord.z() < mD.z())
		{
			do
			{
				cubeG = ccc.getAndBlockCube(id);
			}
			while(cubeG == 0);

			ccc.unlockCube(id);
		}
		
		#ifndef DISK_TIMING 
			++show_progress;
		#endif
	}
}

int main(int argc, char ** argv)
{
	int nLevels = 10;
	int levelCube = 8;
	vmml::vector<3,int> offset(0,0,0);

	lunchbox::Clock clock;


	std::vector<std::string> parameters;
	parameters.push_back(std::string(argv[1]));
	parameters.push_back(std::string(argv[2]));

	if (!cpc.initParameter(parameters))
	{
		std::cerr<<"Error init control plane cache"<<std::endl;
		return 0;
	}
	hdf5File.init(parameters);
	if (!ccc.initParameter(&cpc, eqMivt::getBestDevice()))
	{
		std::cerr<<"Error init control cube cache"<<std::endl;
	}

	vmml::vector<3, int> dim = hdf5File.getRealDimension();

	std::cout<<"Checking errors........."<<std::endl;

	bool error = false;
	for(int i=0; i<10 && !error; i++)
	{
		vmml::vector<3,int> s;
		vmml::vector<3,int> e;
		int nLevels = 0;
		int dimA = 0;
		int dimV = 0;
		do
		{
			s.set(rand() % dim.x(), 0, rand() % dim.z());
			do
			{
				e.set(rand() % (dim.x() - s.x()) + s.x(), rand() % (dim.y() - s.y()) + s.y(), rand() % (dim.z() - s.z()) + s.z());
			}
			while(s.x() >= e.x() || s.y() >= e.y() || s.z() >= e.z());
			 
			/* Calcular dimension del árbol*/
			dimA = fmin(e.x()-s.x(), fmin(e.y() - s.y(), e.z() - s.z()));
			float aux = logf(dimA)/logf(2.0);
			float aux2 = aux - floorf(aux);
			nLevels = aux2>0.0 ? aux+1 : aux;
			dimV = exp2(nLevels);
		}
		while(nLevels <= 1 || s.x()+dimV >= dim.x() || s.y()+dimV >= dim.y() || s.z()+dimV >= dim.z());

		int levelCube = rand() % (nLevels - 1) + 1;

		std::cout<<"Test "<<i<<" nLevels "<<nLevels<<" levelCube "<<levelCube<<" dimension "<<exp2(nLevels - levelCube)<<" offset "<<s<<" : "<<std::endl;

		error = test(nLevels, levelCube, s);
		if (error)
			std::cout<<"Test Fail!"<<std::endl;
		else
		{
			std::cout<<"Test OK"<<std::endl;
		}
	}

	std::cout<<"Checking performance........."<<std::endl;

	for(int i=0; i<10 && !error; i++)
	{
		vmml::vector<3,int> s;
		vmml::vector<3,int> e;
		int nLevels = 0;
		int dimA = 0;
		int dimV = 0;
		do
		{
			s.set(rand() % dim.x(), 0, rand() % dim.z());
			do
			{
				e.set(rand() % (dim.x() - s.x()) + s.x(), rand() % (dim.y() - s.y()) + s.y(), rand() % (dim.z() - s.z()) + s.z());
			}
			while(s.x() >= e.x() || s.y() >= e.y() || s.z() >= e.z());
			 
			dimA = fmin(e.x()-s.x(), fmin(e.y() - s.y(), e.z() - s.z()));;
			/* Calcular dimension del árbol*/
			float aux = logf(dimA)/logf(2.0);
			float aux2 = aux - floorf(aux);
			nLevels = aux2>0.0 ? aux+1 : aux;
			dimV = exp2(nLevels);
		}
		while(nLevels <= 1 || s.x()+dimV >= dim.x() || s.y()+dimV >= dim.y() || s.z()+dimV >= dim.z());

		int levelCube = rand() % (nLevels - 1) + 1;

		std::cout<<"Test "<<i<<" nLevels "<<nLevels<<" levelCube "<<levelCube<<" dimension "<<exp2(nLevels - levelCube)<<" offset "<<s<<" : "<<std::endl;

		double time = 0.0;
		clock.reset();
		testPerf(nLevels, levelCube, s);
		time = clock.getTimed()/1000.0;
		double bw = ((((dimV-s.x())*(dimV-s.y())*(dimV-s.z()))*sizeof(float))/1204.0/1024.0)/time;

		std::cout<<"Test "<<s<<" "<<e<<": "<<time<<" seconds ~ "<<bw<<" MB/s"<<std::endl;
	}

	if (!error)
	{
		int dimA = fmax(dim.x(), fmaxf(dim.y(), dim.z()));
		nLevels = 0;
		/* Calcular dimension del árbol*/
		float aux = logf(dimA)/logf(2.0);
		float aux2 = aux - floorf(aux);
		nLevels = aux2>0.0 ? aux+1 : aux;

		levelCube = rand() % (nLevels - 4) + 4;
		std::cout<<"Test reading complete volume"<<std::endl;
		if (test(nLevels, levelCube, vmml::vector<3,int>(0,0,0)))
		{
			std::cerr<<"Test Fail!!"<<std::endl;	
		}
		else
		{
			double time = 0.0;
			clock.reset();
			testPerf(nLevels, levelCube, vmml::vector<3,int>(0,0,0));
			time = clock.getTimed()/1000.0;
			double bw = (((dim.x()*dim.y()*dim.z())*sizeof(float))/1204.0/1024.0)/time;

			std::cout<<"Read complete volume "<<dim<<" : "<<time<<" seconds ~ "<<bw<<" MB/s"<<std::endl; 
		}
	}


	ccc.stopWork();
	cpc.stopWork();
	hdf5File.close();

	return 0;
}
