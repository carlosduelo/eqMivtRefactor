/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <octreeManager.h>
#include <octree_cuda.h>
#include <visibleCubes.h>
#include <cuda_help.h>
#include <cache.h>
#include <cacheManager.h>
#include <rayCaster_cuda.h>

#include <FreeImage.h>

#include <cmath>

#include <vmmlib/matrix.hpp>

#include <lunchbox/clock.h>

eqMivt::OctreeManager oM;
eqMivt::VisibleCubes vc;
eqMivt::CacheManager cM;
	
float * colors = 0;
float * r = 0;
float * g = 0;
float * b = 0;

int device = 0;

vmml::vector<3, int> realDimVolume;


bool testCubes()
{
	for(int f=0; f<oM.getNumOctrees(); f++)
	{
		eqMivt::Octree * o = oM.getOctree(device);

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

		/* LAUNG OCTREE */
		eqMivt::getBoxIntersectedOctree(o->getOctree(), o->getSizes(), o->getnLevels(),
										VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
										w, h, pvpW, pvpH, o->getmaxLevel(), vc.getSizeGPU(),
										vc.getVisibleCubesGPU(), vc.getIndexVisibleCubesGPU(),
										o->getxGrid(), o->getyGrid(), o->getzGrid(), VectorToInt3(o->getRealDim()), 0);

		/* LAUNCH OCTREE */

		/* Ray Casting */
		eqMivt::rayCasterCubes(	VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
								w, h, pvpW, pvpH, vc.getSizeGPU(), o->getmaxLevel(), o->getnLevels(),
								vc.getVisibleCubesGPU(), vc.getIndexVisibleCubesGPU(), 
								o->getMaxHeight(), buffer, o->getxGrid(), o->getyGrid(), o->getzGrid(),
								VectorToInt3(o->getRealDim()), r, g, b, 0);
		/* Ray Casting */

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
		name<<"frameCubes"<<f<<".png";
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

bool test()
{
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
									VectorToInt3(o->getRealDim()), r, g, b, 0);
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

int main(int argc, char ** argv)
{
	std::vector<std::string> parameters;
	parameters.push_back(std::string(argv[1]));
	parameters.push_back(std::string(argv[2]));

	device = eqMivt::getBestDevice();

	if (!cM.init(parameters))
	{
		std::cerr<<"Error init control plane cache"<<std::endl;
		return 0;
	}

	if (!oM.init(argv[3]))
	{
		std::cerr<<"Error init octree container"<<std::endl;
		return 0;
	}

	FreeImage_Initialise();

	// Init colors
	float * _colorsC = new float[3*NUM_COLORS + 3];

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

	if (cudaSuccess != cudaMalloc((void**)&colors, (3*NUM_COLORS + 3)*sizeof(float)))
	{
		std::cerr<<"Error creating colors"<<std::endl;
		return 0;
	}
	if (cudaSuccess != cudaMemcpy((void*)colors, (void*)_colorsC, (3*NUM_COLORS + 3)*sizeof(float), cudaMemcpyHostToDevice))
	{
		std::cerr<<"Error creating colors"<<std::endl;
		return 0;
	}
	r = colors;
	g = colors + NUM_COLORS + 1;
	b = colors + 2*(NUM_COLORS + 1);

	delete[] _colorsC;

	realDimVolume = oM.getRealDimVolume();

	std::cout<<"============ Creating cube pictures ============"<<std::endl;
	if (!testCubes())
	{
		std::cout<<"Test Fail"<<std::endl;
	}
	for(int f=0; f<oM.getNumOctrees(); f++)
	{
		if (!oM.loadPreviusIsosurface())
			oM.loadPreviusPosition();
	}

	std::cout<<"============ Creating pictures ============"<<std::endl;

	if (test())
	{
		std::cout<<"Test ok"<<std::endl;
	}
	else
	{
		std::cout<<"Test Fail"<<std::endl;
	}

	FreeImage_DeInitialise();

	oM.stop();
	cM.stop();
	vc.destroy();
	cudaFree(colors);
}
