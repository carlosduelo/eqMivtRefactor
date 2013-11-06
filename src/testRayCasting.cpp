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

#include <boost/lexical_cast.hpp>

eqMivt::ResourcesManager rM;
eqMivt::RenderPNG	render;
float mO = 1.0f;
	
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
		
		vmml::vector<3, float> startV = rM.getGridStartCoord();
		vmml::vector<3, float> endV = rM.getGridEndCoord();
	
		float xp = (endV.x() - startV.x())/2.0f;
		float xy = (endV.y() - startV.y())/2.0f;
		float dZ = xp/tan(fov*M_PI/180);


		vmml::vector<4, float> origin(startV.x() + xp, xy, endV.z() + dZ, 1.0f);
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

	std::string colorF = "";

	if (argc == 5)
	{
		try
		{
			std::string n(argv[4]);
			mO = boost::lexical_cast<double>(n);
		}
		catch(...)
		{
			colorF = argv[4];
		}
	}
	else if (argc == 6)
	{
		try
		{
			std::string n(argv[5]);
			mO = boost::lexical_cast<double>(n);
		}
		catch(...)
		{
			colorF = argv[4];
		}
	}

	if (!rM.init(parameters, argv[3], colorF, mO))
	{
		std::cerr<<"Error init resources manager"<<std::endl;
		return 0;
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
