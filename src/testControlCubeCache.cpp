/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlCubeCache.h>

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

	std::cout<<1<< " dir "<<ccc.getAndBlockCube(1)<<std::endl;

	lunchbox::sleep(5000);	

	std::cout<<1<< " dir "<<ccc.getAndBlockCube(1)<<std::endl;


	ccc.stopProcessing();
	cpc.stopProcessing();

	return 0;
}
