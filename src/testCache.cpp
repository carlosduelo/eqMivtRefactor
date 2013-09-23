/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <cache.h>

#include <mortonCodeUtil_CPU.h>
#include <testVisibleCubes_CUDA.h>

#include <lunchbox/sleep.h>
#include <boost/progress.hpp>

#define MIN_RESOLUTION 800*600
#define MAX_RESOLUTION 1920*1080

eqMivt::ControlPlaneCache	cpc;
eqMivt::ControlCubeCache	ccc;
eqMivt::hdf5File hdf5File;
vmml::vector<3,int> mD;

void test(int nLevels, int levelCube, vmml::vector<3,int> offset, int rayCastingLevel, int numPixels)
{
	int dimV = exp2(nLevels);
	vmml::vector<3,int> sP;
	vmml::vector<3,int> eP;

	sP[0] = offset.x() - CUBE_INC < 0 ? 0 : offset.x() - CUBE_INC; 
	sP[1] = offset.y() - CUBE_INC < 0 ? 0 : offset.y() - CUBE_INC; 
	sP[2] = offset.z() - CUBE_INC < 0 ? 0 : offset.z() - CUBE_INC; 
	eP[0] = offset.x() + dimV + CUBE_INC >= mD.x() ? mD.x() : offset.x() + dimV + CUBE_INC;
	eP[1] = offset.y() + dimV + CUBE_INC >= mD.y() ? mD.y() : offset.y() + dimV + CUBE_INC;
	eP[2] = offset.z() + dimV + CUBE_INC >= mD.z() ? mD.z() : offset.z() + dimV + CUBE_INC;

	std::cout<<"ReSize Plane Cache "<<sP<<" "<<eP<<std::endl;
	std::cout<<"Subset volume "<<offset - vmml::vector<3,int>(CUBE_INC,CUBE_INC,CUBE_INC)<<" "<<offset+vmml::vector<3,int>(dimV+CUBE_INC, dimV+CUBE_INC,dimV+CUBE_INC)<<std::endl;

	cpc.reSize(sP, eP);

	ccc.reSize(nLevels, levelCube, offset);

	eqMivt::VisibleCubes		vc;
	vc.reSize(numPixels);
	vc.init();

	eqMivt::Cache				cache;
	cache.init(&ccc);
	cache.setRayCastingLevel(rayCastingLevel);

	eqMivt::index_node_t idS = eqMivt::coordinateToIndex(vmml::vector<3,int>(0,0,0), rayCastingLevel,nLevels);
	eqMivt::index_node_t idE = eqMivt::coordinateToIndex(vmml::vector<3,int>(dimV-1, dimV-1, dimV-1), rayCastingLevel,nLevels);
		
	boost::progress_display show_progress(numPixels);
	int rest = 0;

	while(vc.getListCubes(PAINTED).size() != numPixels)
	{
		// RANDOM FROM NOCUBE TO CUBE OR NOCUBE AND SET RANADM ID
		std::vector<eqMivt::visibleCube_t> changes;
		std::vector<int> l = vc.getListCubes(NOCUBE);
		#if 0
		for(std::vector<int>::iterator it = l.begin(); it!=l.end(); it++)
		{
			eqMivt::visibleCube_t c = vc.getCube(*it);
			c.state = rand() % 2 == 1 ? CUBE : NOCUBE; 
			c.id = (rand()%(idE-idS))+idS;
			changes.push_back(c);
		}
		vc.updateVisibleCubes(changes);
		#endif
		vc.updateGPU(NOCUBE, false, 0);
		test_randomNOCUBE_To_NOCUBEorCUBE(vc.getVisibleCubesGPU(), vc.getSizeGPU(), idS, idE);
		vc.updateCPU();

		changes.clear();

		// CACHED PUSH
		cache.pushCubes(&vc);

		// (RANDOM FROM CACHED TO NOCUBE OR PAINTED) AND (FROM NOCUBE TO PAINTED) 
		l = vc.getListCubes(CACHED);
		for(std::vector<int>::iterator it = l.begin(); it!=l.end(); it++)
		{
			eqMivt::visibleCube_t c = vc.getCube(*it);
			c.state = rand() % 2 == 1 ? NOCUBE : PAINTED; 
			changes.push_back(c);
		}
		l = vc.getListCubes(NOCUBE);
		for(std::vector<int>::iterator it = l.begin(); it!=l.end(); it++)
		{
			eqMivt::visibleCube_t c = vc.getCube(*it);
			c.state = PAINTED; 
			changes.push_back(c);
		}
		vc.updateVisibleCubes(changes);
		changes.clear();

		//CACHED POP
		cache.popCubes();

		show_progress += vc.getNumElements(PAINTED) - rest;
		rest += vc.getNumElements(PAINTED) - rest;
	}
}

int main(int argc, char ** argv)
{
	std::vector<std::string> parameters;
	parameters.push_back(std::string(argv[1]));
	parameters.push_back(std::string(argv[2]));

	cpc.initParameter(parameters);
	ccc.initParameter(&cpc);

	hdf5File.init(parameters);
	mD = hdf5File.getRealDimension();
	hdf5File.close();

	std::cout<<"Volume size "<<mD<<std::endl;

	cpc.start();
	ccc.start();

	for(int i=0; i<100; i++)
	{
		vmml::vector<3,int> s;
		vmml::vector<3,int> e;
		int nLevels = 0;
		do
		{
			s.set(rand() % mD.x(), 0, rand() % mD.z());
			do
			{
				e.set(rand() % (mD.x() - s.x()) + s.x(), rand() % (mD.y() - s.y()) + s.y(), rand() % (mD.z() - s.z()) + s.z());
			}
			while(s.x() >= e.x() || s.y() >= e.y() || s.z() >= e.z());
		 
			/* Calcular dimension del árbol*/
			int dim = fmin(e.x()-s.x(), fmin(e.y() - s.y(), e.z() - s.z()));;
			float aux = logf(dim)/logf(2.0);
			float aux2 = aux - floorf(aux);
			nLevels = aux2>0.0 ? aux+1 : aux;
		}
		while(nLevels <= 0);

		int levelCube = rand() % (nLevels - 1) + 1;
		int rayLevel = rand() % (nLevels - levelCube) + levelCube;
		int numPixels = (rand() % (MAX_RESOLUTION - MIN_RESOLUTION)) + MIN_RESOLUTION;

		std::cout<<"Test "<<i<<" pixels "<<numPixels<<" nLevels "<<nLevels<<" levelCube "<<levelCube<<" dimension "<<exp2(nLevels - levelCube)<<" offset "<<s<<" : "<<std::endl;
		test(nLevels, levelCube, s, rayLevel, numPixels);
	}


	ccc.stopProcessing();
	cpc.stopProcessing();

	return 0;

}
