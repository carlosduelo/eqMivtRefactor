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


	lunchbox::sleep(5000);	

	for(int i=0; i<4; i++)
	{
		float * data = cpc.getAndBlockPlane(i);
		std::cout<<" get and lock plane "<<i<<" direcion "<<data<<std::endl;
	}

	lunchbox::sleep(5000);	

	for(int i=4; i<14; i++)
	{
		float * data = cpc.getAndBlockPlane(i);
		std::cout<<" get and lock plane "<<i<<" direcion "<<data<<std::endl;
	}

	cpc.unlockPlane(1);

	lunchbox::sleep(5000);	

	for(int i=3; i<14; i++)
	{
		float * data = cpc.getAndBlockPlane(i);
		std::cout<<" get and lock plane "<<i<<" direcion "<<data<<std::endl;
	}

	cpc.unlockPlane(0);
	lunchbox::sleep(5000);	

	for(int i=3; i<14; i++)
	{
		float * data = cpc.getAndBlockPlane(i);
		std::cout<<" get and lock plane "<<i<<" direcion "<<data<<std::endl;
	}

	cpc.unlockPlane(2);
	lunchbox::sleep(5000);	
	for(int i=3; i<14; i++)
	{
		float * data = cpc.getAndBlockPlane(i);
		std::cout<<" get and lock plane "<<i<<" direcion "<<data<<std::endl;
	}
	for(int i=0; i<4; i++)
	{
		float * data = cpc.getAndBlockPlane(i);
		std::cout<<" get and lock plane "<<i<<" direcion "<<data<<std::endl;
	}

	lunchbox::sleep(5000);	

	cpc.stopProcessing();

	return 0;
}
