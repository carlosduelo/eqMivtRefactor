/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <octree.h>
#include <octree_cuda.h>
#include <visibleCubes.h>
#include <cuda_help.h>

#include <FreeImage.h>

#include <cmath>

#include <vmmlib/matrix.hpp>

int main(int argc, char ** argv)
{
	eqMivt::OctreeContainer oC;
	eqMivt::Octree o;
	eqMivt::VisibleCubes vc;

	if (!oC.init(argv[1]))
	{
		std::cerr<<"Error init octree container"<<std::endl;
		return 0;
	}
	if (!o.init(&oC, eqMivt::getBestDevice()))
	{
		std::cerr<<"Error init octree"<<std::endl;
		return 0;
	}

	FreeImage_Initialise();

	for(int f=0; f<oC.getNumOctrees(); f++)
	{
		o.loadCurrentOctree();

		int pvpW = 1024;
		int pvpH = 1024;
		int numPixels = pvpW*pvpH;
		float tnear = 1.0f;
		float fov = 30;
		
		vmml::vector<3, float> startV = o.getGridStartCoord();
		vmml::vector<3, float> endV = o.getGridEndCoord();
	
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

		/* LAUNG OCTREE */
		vc.reSize(numPixels);
		vc.init();
		eqMivt::getBoxIntersectedOctree(o.getOctree(), o.getSizes(), o.getnLevels(),
										VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
										w, h, pvpW, pvpH, o.getmaxLevel(), vc.getSizeGPU(),
										vc.getVisibleCubesGPU(), vc.getIndexVisibleCubesGPU(),
										o.getxGrid(), o.getyGrid(), o.getzGrid(), VectorToInt3(o.getRealDim()), 0);

		vc.updateCPU();
		/* LAUNCH OCTREE */

		std::cout<<"No hit: "<<vc.getNumElements(NOCUBE)<<" hit: "<<vc.getNumElements(CUBE)<<std::endl;

		/* CREATE PNG */
		FIBITMAP * bitmap = FreeImage_Allocate(pvpW, pvpH, 24);
		RGBQUAD color;

		for(int i=0; i<pvpH; i++)
		{
			for(int j=0; j<pvpH; j++)
			{
				int id = i*pvpW+ j;
				if (vc.getCube(id)->state == CUBE)
				{
					color.rgbRed	= 255;
					color.rgbGreen	= 0;
					color.rgbBlue	= 0;
				}
				else if(vc.getCube(id)->state == NOCUBE) 
				{
					color.rgbRed	= 255;
					color.rgbGreen	= 255;
					color.rgbBlue	= 255;
				}
				else
				{
					color.rgbRed	= 0;
					color.rgbGreen	= 255;
					color.rgbBlue	= 0;
				}
				FreeImage_SetPixelColor(bitmap, j, i, &color);
			}
		}
		std::stringstream name;
		name<<"frame"<<f<<".png";
		FreeImage_Save(FIF_PNG, bitmap, name.str().c_str(), 0);

		delete bitmap;
		/* END CREATE PNG */

		if (!oC.loadNextIsosurface())
			oC.loadNextPosition();
	}

	FreeImage_DeInitialise();

	o.stop();
	oC.stop();
	vc.destroy();
}
