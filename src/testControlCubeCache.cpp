/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlCubeCache.h>

#include <mortonCodeUtil_CPU.h>

#include <lunchbox/sleep.h>

int main(int argc, char ** argv)
{
	eqMivt::ControlPlaneCache cpc;
	std::vector<std::string> parameters;
	parameters.push_back(std::string(argv[1]));
	parameters.push_back(std::string(argv[2]));
	cpc.initParameter(parameters, 0);
	cpc.start();

	eqMivt::ControlCubeCache ccc;
	ccc.initParameter(&cpc);
	ccc.start();

	ccc.reSize(9,9,vmml::vector<3,int>(0,0,0));

	lunchbox::sleep(5000);	

	ccc.reSize(10,10,vmml::vector<3,int>(0,0,0));

	lunchbox::sleep(5000);

	eqMivt::index_node_t id = eqMivt::coordinateToIndex(vmml::vector<3, int>(0,0,0), 10, 10);

	std::cout<<id<< " dir "<<ccc.getAndBlockCube(id)<<std::endl;

	lunchbox::sleep(5000);	

	std::cout<<id<< " dir "<<ccc.getAndBlockCube(id)<<std::endl;

	ccc.unlockCube(id);

	ccc.stopProcessing();
	cpc.stopProcessing();

	return 0;
}
