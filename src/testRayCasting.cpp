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
#if 0

bool test()
{
	eqMivt::color_t c = coM.getColors(device);

	for(int f=0; f<oM.getNumOctrees(); f++)
	{
		if (!cM.freeMemoryAndPause())
		{
			std::cerr<<"Error, free plane cache"<<std::endl;
			return false;
		}

		// Load in gpu octree
		eqMivt::Octree * o = oM.getOctree(device);

		// Resize cube cache and plane cache
		vmml::vector<3, int> sP, eP;
		sP[0] = o->getStartCoord().x() - CUBE_INC < 0 ? 0 : o->getStartCoord().x() - CUBE_INC; 
		sP[1] = o->getStartCoord().y() - CUBE_INC < 0 ? 0 : o->getStartCoord().y() - CUBE_INC; 
		sP[2] = o->getStartCoord().z() - CUBE_INC < 0 ? 0 : o->getStartCoord().z() - CUBE_INC; 
		eP[0] = o->getEndCoord().x() + CUBE_INC >= realDimVolume.x() ? realDimVolume.x() : o->getEndCoord().x() + CUBE_INC;
		eP[1] = o->getEndCoord().y() + CUBE_INC >= realDimVolume.y() ? realDimVolume.y() : o->getEndCoord().y() + CUBE_INC;
		eP[2] = o->getEndCoord().z() + CUBE_INC >= realDimVolume.z() ? realDimVolume.z() : o->getEndCoord().z() + CUBE_INC;

		if (!cM.reSizeAndContinue(sP, eP, o->getnLevels(), o->getCubeLevel(), o->getStartCoord()))
		{
			std::cerr<<"Error, resizing plane cache"<<std::endl;
			return false;
		}
		// Create Cache
		eqMivt::Cache				cache;
		if (!cache.init(cM.getCubeCache(device)))
		{
			std::cerr<<"Error, creating cache in device"<<std::endl;
			return false;
		}
		
		cache.setRayCastingLevel(o->getRayCastingLevel());

		int pvpW = 1024;
		int pvpH = 1024;
		int numPixels = pvpW*pvpH;

		// SCREEN AND COLORS
		float * bufferC = new float[3*numPixels];
		float * buffer = 0;
		if (cudaSuccess != cudaMalloc((void**)&buffer, 3*numPixels*sizeof(float)))
		{
			std::cerr<<"Error allocating pixel buffer"<<std::endl;
			return 0;
		}

		float tnear = 1.0f;
		float fov = 30;
		
		vmml::vector<3, int> startV = o->getStartCoord();
		vmml::vector<3, int> endV = o->getEndCoord();
		//vmml::vector<4, float> origin(36 , 38, 2, 1.0f);
		vmml::vector<4, float> origin(startV.x() + ((endV.x()-startV.x())/3.0f), o->getMaxHeight(), 1.1f*endV.z(), 1.0f);
		//vmml::vector<4, float> origin( 0, 0, 1.1f*endV.z(), 1.0f);
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

		vc.reSize(numPixels);
		vc.init();

		lunchbox::Clock clockOperation;
		lunchbox::Clock clockTotal;
		double oT = 0.0; 
		double cT = 0.0; 
		double rT = 0.0; 
		clockTotal.reset();
		while(vc.getNumElements(PAINTED) != vc.getSize())
		{
			clockOperation.reset();
			/* LAUNG OCTREE */
			vc.updateGPU(NOCUBE, 0);

			eqMivt::getBoxIntersectedOctree(o->getOctree(), o->getSizes(), o->getnLevels(),
											VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
											w, h, pvpW, pvpH, o->getRayCastingLevel(), vc.getSizeGPU(),
											vc.getVisibleCubesGPU(), vc.getIndexVisibleCubesGPU(),
											o->getxGrid(), o->getyGrid(), o->getzGrid(), VectorToInt3(o->getRealDim()), 0);

			vc.updateCPU();
			#if 0
			int t = vc.getNumElements(NOCUBE) + vc.getNumElements(CUBE) + vc.getNumElements(CACHED) + vc.getNumElements(PAINTED);
			std::cout<<"Octree--> NOCUBE: "<<vc.getNumElements(NOCUBE)<<" CUBE: "<<vc.getNumElements(CUBE)<<" CACHED: "<<vc.getNumElements(CACHED)<<" PAINTED: "<<vc.getNumElements(PAINTED)<<" "<<t<<" "<<vc.getSize()<<std::endl;
			#endif
			/* LAUNCH OCTREE */
			oT+=clockOperation.getTimed()/1000.0;

			clockOperation.reset();
			/* Lock Cubes*/
			cache.pushCubes(&vc);
			vc.updateIndexCPU();
			/* Lock Cubes*/
			cT += clockOperation.getTimed()/1000.0;

			#if 0
			t = vc.getNumElements(NOCUBE) + vc.getNumElements(CUBE) + vc.getNumElements(CACHED) + vc.getNumElements(PAINTED);
			std::cout<<"Cache--> NOCUBE: "<<vc.getNumElements(NOCUBE)<<" CUBE: "<<vc.getNumElements(CUBE)<<" CACHED: "<<vc.getNumElements(CACHED)<<" PAINTED: "<<vc.getNumElements(PAINTED)<<" "<<t<<" "<<vc.getSize()<<std::endl;
			#endif
		
			clockOperation.reset();
			/* Ray Casting */
			vc.updateGPU(NOCUBE | CACHED, 0);
			eqMivt::rayCaster(	VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
									w, h, pvpW, pvpH, vc.getSizeGPU(), o->getRayCastingLevel(), o->getCubeLevel(), o->getnLevels(),
									o->getIsosurface(), vc.getVisibleCubesGPU(), vc.getIndexVisibleCubesGPU(), 
									o->getMaxHeight(), buffer, o->getxGrid(), o->getyGrid(), o->getzGrid(),
									VectorToInt3(o->getRealDim()), c.r, c.g, c.b, 0);
			vc.updateCPU();
			/* Ray Casting */
			rT += clockOperation.getTimed()/1000.0;

			clockOperation.reset();
			/* Unlock Cubes*/
			cache.popCubes();
			/* Unlock Cubes*/
			cT += clockOperation.getTimed()/1000.0;

			#if 0
			t = vc.getNumElements(NOCUBE) + vc.getNumElements(CUBE) + vc.getNumElements(CACHED) + vc.getNumElements(PAINTED);
			std::cout<<"Ray Casting--> NOCUBE: "<<vc.getNumElements(NOCUBE)<<" CUBE: "<<vc.getNumElements(CUBE)<<" CACHED: "<<vc.getNumElements(CACHED)<<" PAINTED: "<<vc.getNumElements(PAINTED)<<" "<<t<<" "<<vc.getSize()<<std::endl;
			#endif
		}
		std::cout<<"Octree elapsed time "<<oT<<" seconds"<<std::endl;
		std::cout<<"Cache elapsed time "<<cT<<" seconds"<<std::endl;
		std::cout<<"RayCaster elapsed time "<<rT<<" seconds"<<std::endl;
		std::cout<<"Total elapsed time "<<clockTotal.getTimed()/1000.0<<" seconds"<<std::endl;

		if (cudaSuccess != cudaMemcpy((void*)bufferC, (void*)buffer, 3*numPixels*sizeof(float), cudaMemcpyDeviceToHost))
		{
			std::cerr<<"Error copying pixel buffer to cpu"<<std::endl;
			return 0;
		}
		/* CREATE PNG */
		FIBITMAP * bitmap = FreeImage_Allocate(pvpW, pvpH, 24);
		RGBQUAD color;

		for(int i=0; i<pvpH; i++)
		{
			for(int j=0; j<pvpH; j++)
			{
				int id = i*pvpW+ j;
				color.rgbRed	= bufferC[id*3]*255;
				color.rgbGreen	= bufferC[id*3+1]*255;
				color.rgbBlue	= bufferC[id*3+2]*255;
				FreeImage_SetPixelColor(bitmap, j, i, &color);
			}
		}
		std::stringstream name;
		name<<"frame"<<f<<".png";
		FreeImage_Save(FIF_PNG, bitmap, name.str().c_str(), 0);

		delete bitmap;
		/* END CREATE PNG */

		if (!oM.loadNextIsosurface())
			oM.loadNextPosition();

		delete[] bufferC;
		if (cudaSuccess != cudaFree(buffer))
		{
			std::cerr<<"Error free pixel buffer"<<std::endl;
		}

	}

	return true;
}
#endif

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
		if (!rM.init(parameters, argv[3], argv[3]))
		{
			std::cerr<<"Error init resources manager"<<std::endl;
			return 0;
		}
	}
	else
	{
		if (!rM.init(parameters, argv[3], ""))
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
