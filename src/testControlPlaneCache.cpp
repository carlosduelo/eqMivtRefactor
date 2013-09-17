/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlPlaneCache.h>

#include <lunchbox/sleep.h>

int main(int argc, char ** argv)
{
	eqMivt::ControlPlaneCache cpc;
	std::vector<std::string> parameters;
	parameters.push_back(std::string(argv[1]));
	parameters.push_back(std::string(argv[2]));
	cpc.initParamenter(parameters, 0);

	cpc.start();

	for(int i=0; i<4; i++)
	{
		std::cout<<" get and lock plane "<<i<<" direcion "<<cpc.getAndBlockPlane(i)<<std::endl;
	}


	lunchbox::sleep(20000);	

	for(int i=0; i<4; i++)
	{
		std::cout<<" get and lock plane "<<i<<" direcion "<<cpc.getAndBlockPlane(i)<<std::endl;
	}

	lunchbox::sleep(40000);	

#if 0
	for(int i=4; i<14; i++)
	{
		cpc.addPlane(i);
		if (i % 7 == 0)
		cpc.getAndBlockPlane(i);
	}

	cpc.unlockPlane(1);

	lunchbox::sleep(20000);	
	for(int i=0; i<4; i++)
	{
		cpc.unlockPlane(i);
	}
#endif

	cpc.stopProcessing();

	return 0;
}
