/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <cache.h>

#include <mortonCodeUtil_CPU.h>

#include <lunchbox/sleep.h>

eqMivt::ControlPlaneCache	cpc;
eqMivt::ControlCubeCache	ccc;

bool test(int nLevels, int levelCube, vmml::vector<3,int> offset, int rayCastingLevel, int numPixels)
{
	ccc.reSize(nLevels, levelCube, offset);

	eqMivt::VisibleCubes		vc;
	vc.reSize(numPixels);
	vc.init();

	eqMivt::Cache				cache;
	cache.init(&ccc);
	cache.setRayCastingLevel(rayCastingLevel);

	int dim = exp2(nLevels);
	eqMivt::index_node_t idS = eqMivt::coordinateToIndex(vmml::vector<3,int>(0,0,0), rayCastingLevel,nLevels);
	eqMivt::index_node_t idE = eqMivt::coordinateToIndex(vmml::vector<3,int>(dim-1, dim-1, dim-1), rayCastingLevel,nLevels);

	while(vc.getListCubes(PAINTED).size() != numPixels)
	{
		// RANDOM FROM NOCUBE TO CUBE OR NOCUBE AND SET RANADM ID
		std::vector<eqMivt::visibleCube_t> changes;
		std::vector<int> l = vc.getListCubes(NOCUBE);
		for(std::vector<int>::iterator it = l.begin(); it!=l.end(); it++)
		{
			eqMivt::visibleCube_t c = vc.getCube(*it);
			c.state = rand() % 2 == 1 ? CUBE : NOCUBE; 
			c.id = (rand()%(idE-idS))+idS;;
			changes.push_back(c);
		}
		vc.updateVisibleCubes(changes);
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
	}


	return false;
}

int main(int argc, char ** argv)
{
	std::vector<std::string> parameters;
	parameters.push_back(std::string(argv[1]));
	parameters.push_back(std::string(argv[2]));

	cpc.initParameter(parameters, vmml::vector<3,int>(0,0,0),vmml::vector<3,int>(0,0,0));
	ccc.initParameter(&cpc);

	cpc.start();
	ccc.start();

	lunchbox::sleep(500);

	bool error =	!test(2, 2, vmml::vector<3,int>(0,0,0),2,100) && 
					!test(10, 1, vmml::vector<3,int>(0,0,0),1,100) &&
					!test(10, 2, vmml::vector<3,int>(0,0,0),2,100) &&
					!test(10, 3, vmml::vector<3,int>(0,0,0),3,100) &&
					!test(10, 4, vmml::vector<3,int>(0,0,0),4,100) &&
					!test(10, 5, vmml::vector<3,int>(0,0,0),5,100) &&
					!test(10, 6, vmml::vector<3,int>(0,0,0),6,100) &&
					!test(10, 10, vmml::vector<3,int>(0,0,0),8,100) &&
					!test(2, 2, vmml::vector<3,int>(255,0,123),1,100) &&
					!test(10, 1, vmml::vector<3,int>(123,0,0),1,100) &&
					!test(10, 2, vmml::vector<3,int>(321,0,2),2,100) &&
					!test(10, 3, vmml::vector<3,int>(12,0,300),2,100) &&
					!test(10, 4, vmml::vector<3,int>(42,0,99),4,100) &&
					!test(10, 5, vmml::vector<3,int>(12,0,100),5,100) &&
					!test(10, 6, vmml::vector<3,int>(50,0,30),5,100) &&
					!test(10, 10, vmml::vector<3,int>(60,0,0),8,100) &&
					!test(10, 10, vmml::vector<3,int>(90,0,12),8,100);


	ccc.stopProcessing();
	cpc.stopProcessing();

	lunchbox::sleep(5000);

	if (error)
		std::cout<<"Test Fail"<<std::endl;
	else
		std::cout<<"Test OK"<<std::endl;

	return 0;

}
