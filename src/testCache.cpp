/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <cache.h>

#include <mortonCodeUtil_CPU.h>
#include <testVisibleCubes_cuda.h>

#include <lunchbox/sleep.h>
#include <lunchbox/clock.h>
#include <boost/progress.hpp>

#define MIN_RESOLUTION 800*600
#define MAX_RESOLUTION 1920*1080

eqMivt::ControlPlaneCache	cpc;
eqMivt::ControlCubeCache	ccc;
eqMivt::hdf5File hdf5File;
vmml::vector<3,int> mD;
eqMivt::VisibleCubes		vc;

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

	vc.reSize(numPixels);
	vc.init();

	eqMivt::Cache				cache;
	cache.init(&ccc);
	cache.setRayCastingLevel(rayCastingLevel);

	cpc.reSize(sP, eP);
	ccc.reSize(nLevels, levelCube, offset);

	eqMivt::index_node_t idS = eqMivt::coordinateToIndex(vmml::vector<3,int>(0,0,0), rayCastingLevel,nLevels);
	eqMivt::index_node_t idE = eqMivt::coordinateToIndex(vmml::vector<3,int>(dimV-1, dimV-1, dimV-1), rayCastingLevel,nLevels);
		
	boost::progress_display show_progress(numPixels);
	int rest = 0;

	lunchbox::Clock clock;

	double tUpdateGPU = 0.0;
	double tUdateCPU = 0.0;
	double tPush = 0.0;
	double tUpdateVisibleCubes = 0.0;
	double tPop = 0.0;
	double tgetlistCubes = 0.0;
	int iterations = 0;

	while(vc.getListCubes(PAINTED).size() != numPixels)
	{
		// RANDOM FROM NOCUBE TO CUBE OR NOCUBE AND SET RANADM ID
		std::vector<int> l;
	
		clock.reset();
		vc.updateGPU(NOCUBE, 0);
		tUpdateGPU += clock.getTimed()/1000.0;

		test_randomNOCUBE_To_NOCUBEorCUBE(vc.getVisibleCubesGPU(), vc.getIndexVisibleCubesGPU(), vc.getSizeGPU(), idS, idE);

		clock.reset();
		vc.updateCPU();
		tUdateCPU += clock.getTimed()/1000.0;

		// CACHED PUSH
		clock.reset();
		cache.pushCubes(&vc);
		tPush += clock.getTimed()/1000.0;

		// (RANDOM FROM CACHED TO NOCUBE OR PAINTED) AND (FROM NOCUBE TO PAINTED) 
		clock.reset();
		l = vc.getListCubes(CACHED);
		tgetlistCubes += clock.getTimed()/1000.0;

		for(std::vector<int>::iterator it = l.begin(); it!=l.end(); it++)
		{
			eqMivt::visibleCube_t * c = vc.getCube(*it);
			c->state = rand() % 2 == 1 ? NOCUBE : PAINTED; 
		}

		clock.reset();
		l = vc.getListCubes(NOCUBE);
		tgetlistCubes += clock.getTimed()/1000.0;

		for(std::vector<int>::iterator it = l.begin(); it!=l.end(); it++)
		{
			eqMivt::visibleCube_t * c = vc.getCube(*it);
			c->state = PAINTED; 
		}

		clock.reset();
		vc.updateIndexCPU();
		tUpdateVisibleCubes += clock.getTimed()/1000.0;

		//CACHED POP
		clock.reset();
		cache.popCubes();
		tPop += clock.getTimed()/1000.0;

		show_progress += vc.getNumElements(PAINTED) - rest;
		rest += vc.getNumElements(PAINTED) - rest;
		iterations++;
	}

	std::cout<<"Iterations "<<iterations<<std::endl;
	std::cout<<"Time updating visible cubes gpu "<<tUpdateGPU<<" seconds"<<std::endl;
	std::cout<<"Time updating visible cubes cpu "<<tUdateCPU<<" seconds"<<std::endl;
	std::cout<<"Time pushing cubes to cache "<<tPush<<" seconds"<<std::endl;
	std::cout<<"Time geting visible cube list "<<tgetlistCubes<<" seconds"<<std::endl;
	std::cout<<"Time updating visible cubes "<<tUpdateVisibleCubes<<" seconds"<<std::endl;
	std::cout<<"Time poping cubes from cache "<<tPop<<" seconds"<<std::endl;
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

	lunchbox::Clock clock;

	for(int i=0; i<20; i++)
	{
		vmml::vector<3,int> s;
		vmml::vector<3,int> e;
		int nLevels = 0;
		int dimA = 0;
		int dimV = 0;
		do
		{
			s.set(rand() % mD.x(), 0, rand() % mD.z());
			do
			{
				e.set(rand() % (mD.x() - s.x()) + s.x(), rand() % (mD.y() - s.y()) + s.y(), rand() % (mD.z() - s.z()) + s.z());
			}
			while(s.x() >= e.x() || s.y() >= e.y() || s.z() >= e.z());
		 
			/* Calcular dimension del árbol*/
			dimA = fmin(e.x()-s.x(), fmin(e.y() - s.y(), e.z() - s.z()));;
			float aux = logf(dimA)/logf(2.0);
			float aux2 = aux - floorf(aux);
			nLevels = aux2>0.0 ? aux+1 : aux;
			dimV = exp2(nLevels);
		}
		while(nLevels <= 1 ||  s.x()+dimV >= mD.x() || s.y()+dimV >= mD.y() || s.z()+dimV >= mD.z());

		int levelCube = rand() % (nLevels - 1) + 1;
		int rayLevel = rand() % (nLevels - levelCube) + levelCube;
		int numPixels = (rand() % (MAX_RESOLUTION - MIN_RESOLUTION)) + MIN_RESOLUTION;

		std::cout<<"Test "<<i<<" pixels "<<numPixels<<" nLevels "<<nLevels<<" levelCube "<<levelCube<<" dimension "<<exp2(nLevels - levelCube)<<" offset "<<s<<" : "<<std::endl;

		double time = 0.0;
		clock.reset();
		test(nLevels, levelCube, s, rayLevel, numPixels);
		time = clock.getTimed()/1000.0;
		std::cout<<"Time: "<<time<<" seconds"<<std::endl;
	}


	ccc.stopProcessing();
	cpc.stopProcessing();
	vc.destroy();

	return 0;

}
