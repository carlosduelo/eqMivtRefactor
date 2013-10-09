/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <octree.h>
#include <octree_cuda.h>
#include <visibleCubes.h>
#include <cuda_help.h>

#include <cmath>

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

{
	int screenW = 256;
	int screenH = 256;
	int pvpW = 16;
	int pvpH = 16;
	int numPixels = pvpW*pvpH;
	float tnear = 1.0f;

	vc.reSize(numPixels);
	vc.init();


	o.loadCurrentOctree();

	vmml::vector<3, int> startV = o.getStartCoord();
	vmml::vector<3, int> endV = o.getEndCoord();
	vmml::vector<4, float> origin(64,0,400);
	//vmml::vector<4, float> origin(startV.x() + ((endV.x()-startV.x())/2.0f), o.getMaxHeight()/2.0f, 2.0f*endV.z(), 1.0f);
	vmml::vector<4, float> up(0.0f, 1.0f, 0.0f, 0.0f);
	vmml::vector<4, float> right(1.0f, 0.0f, 0.0f, 0.0f);

	vmml::vector<4, float>LB(0.0f, 0.0f, -1.0f, 0.0f); 	
	LB -=  (up*((screenH)/2.0f)); 
	LB -=  right*((screenW)/2.0f); 
	LB.normalize();
	LB = origin + LB*tnear;
	vmml::vector<4, float>LT(0.0f, 0.0f, -1.0f, 0.0f); 	
	LT +=  (up*((screenH)/2.0f)); 
	LT -=  right*((screenW)/2.0f); 
	LT.normalize();
	LT = origin + LT*tnear;
	vmml::vector<4, float>RT(0.0f, 0.0f, -1.0f, 0.0f); 	
	RT +=  (up*((screenH)/2.0f)); 
	RT +=  right*((screenW)/2.0f); 
	RT.normalize();
	RT = origin + RT*tnear;
	vmml::vector<4, float>RB(0.0f, 0.0f, -1.0f, 0.0f); 	
	RB -=  (up*((screenH)/2.0f)); 
	RB +=  right*((screenW)/2.0f); 
	RB.normalize();
	RB = origin + RB*tnear;

	float w = (RB.x() - LB.x())/(float)pvpW;
	float h = (LT.y() - LB.y())/(float)pvpH;

	std::cout<<screenW<<" "<<screenH<<" "<<w<<" "<<h<<std::endl;
	std::cout<<"Camera position "<<origin<<std::endl;
	std::cout<<"Frustum"<<std::endl;
	std::cout<<LB<<std::endl;
	std::cout<<LT<<std::endl;
	std::cout<<RB<<std::endl;
	std::cout<<RT<<std::endl;

	eqMivt::getBoxIntersectedOctree(o.getOctree(), o.getSizes(), o.getnLevels(),
									VectorToFloat3(origin), VectorToFloat3(LB), VectorToFloat3(up), VectorToFloat3(right),
									w, h, pvpW, pvpH, o.getmaxLevel(), numPixels,
									vc.getVisibleCubesGPU(), vc.getIndexVisibleCubesGPU(),
									o.getxGrid(), o.getyGrid(), o.getzGrid(), VectorToInt3(o.getRealDim()), 0);

	vc.updateIndexCPU();

	std::cout<<vc.getNumElements(NOCUBE)<<" "<<vc.getNumElements(CUBE)<<std::endl;
}

	o.stop();
	oC.stop();
	vc.destroy();
}
