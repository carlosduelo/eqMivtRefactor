/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <cuda_help.h>
#include <renderPNG.h>
#include <resourcesManager.h>

#include <vmmlib/matrix.hpp>

#include <lunchbox/clock.h>

eqMivt::ResourcesManager rM;
eqMivt::RenderPNG	render;
	
int device = 0;

bool test(bool cubes)
{
	render.setDrawCubes(cubes);

	for(int f=0; f<rM.getNumOctrees(); f++)
	{
		if (!rM.updateRender(&render))
		{
			std::cerr<<"Error updating render"<<std::endl;
			return false;
		}

		int pvpW = 1024;
		int pvpH = 1024;

		float tnear = 1.0f;
		float fov = 30;
		
		vmml::vector<3, int> startV = rM.getStartCoord();
		vmml::vector<3, int> endV = rM.getEndCoord();
		//vmml::vector<4, float> origin(36 , 38, 2, 1.0f);
		vmml::vector<4, float> origin(startV.x() + ((endV.x()-startV.x())/3.0f), rM.getMaxHeight(), 1.1f*endV.z(), 1.0f);
		vmml::vector<4, float> up(0.0f, 1.0f, 0.0f, 0.0f);
		vmml::vector<4, float> right(1.0f, 0.0f, 0.0f, 0.0f);
		float ft = tan(fov*M_PI/180);
		vmml::vector<4, float>LB(-ft, -ft, -tnear, 1.0f); 	
		vmml::vector<4, float>LT(-ft, ft, -tnear, 1.0f); 	
		vmml::vector<4, float>RT(ft, ft, -tnear, 1.0f); 	
		vmml::vector<4, float>RB(ft, -ft, -tnear, 1.0f); 	

		vmml::matrix<4,4,float> positionM = vmml::matrix<4,4,float>::IDENTITY;
		positionM.set_translation(vmml::vector<3,float>(origin.x(), origin.y(), origin.z()));
		vmml::matrix<4,4,float> model = vmml::matrix<4,4,float>::IDENTITY;
		model = positionM * model;
		LB = model*LB;
		LT = model*LT;
		RB = model*RB;
		RT = model*RT;

		float w = (RB.x() - LB.x())/(float)pvpW;
		float h = (LT.y() - LB.y())/(float)pvpH;

		std::cout<<"Camera position "<<origin<<std::endl;
		std::cout<<"Frustum"<<std::endl;
		std::cout<<LB<<std::endl;
		std::cout<<LT<<std::endl;
		std::cout<<RB<<std::endl;
		std::cout<<RT<<std::endl;

		up = LT-LB;
		right = RB - LB;
		right.normalize();
		up.normalize();

		if (!render.setViewPort(pvpW, pvpH))
		{
			std::cerr<<"Error setting viewport"<<std::endl;
			return false;
		}
		
		if (!render.frameDraw(origin, LB, up, right, w, h))
		{
			std::cerr<<"Error rendering"<<std::endl;
			return false;
		}

		if (f < rM.getNumOctrees()-1 && !rM.loadNext())
		{
			std::cerr<<"Error loading next isosurface"<<std::endl;
			return false;
		}
	}

	return true;
}

int main(int argc, char ** argv)
{
	std::vector<std::string> parameters;
	parameters.push_back(std::string(argv[1]));
	parameters.push_back(std::string(argv[2]));

	device = eqMivt::getBestDevice();

	cudaFuncCache cacheConfig = cudaFuncCachePreferL1;
	if (cudaSuccess != cudaSetDevice(device) || cudaSuccess != cudaDeviceSetCacheConfig(cacheConfig))
	{
		std::cerr<<"Error setting up best device"<<std::endl;
		return 0;
	}

	if (argc == 5)
	{
		if (!rM.init(parameters, argv[3], argv[3], 1.0f))
		{
			std::cerr<<"Error init resources manager"<<std::endl;
			return 0;
		}
	}
	else
	{
		if (!rM.init(parameters, argv[3], "", 1.0f))
		{
			std::cerr<<"Error init resources manager"<<std::endl;
			return 0;
		}
	}
	if (!rM.start())
	{
		std::cerr<<"Error start resources manager"<<std::endl;
		return 0;
	}

	if (!render.init(device,""))
	{
		std::cerr<<"Error init render"<<std::endl;
		return 0;
	}

	
	std::cout<<"============ Creating cube pictures ============"<<std::endl;
	if (!test(true))
	{
		std::cout<<"Test Fail"<<std::endl;
	}

	while(rM.loadPreviusPosition()){};
	while(rM.loadPreviusIsosurface()){};

	std::cout<<"============ Creating pictures ============"<<std::endl;

	if (test(false))
	{
		std::cout<<"Test ok"<<std::endl;
	}
	else
	{
		std::cout<<"Test Fail"<<std::endl;
	}

	render.destroy();
	rM.destroy();

	std::cout<<"End test"<<std::endl;
}
