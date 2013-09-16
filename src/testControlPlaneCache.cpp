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
	cpc.init();
	cpc.start();

	for(int i=0; i<14; i++)
	{
		lunchbox::sleep(2000);	
		cpc.addPlane(i);
	}

	cpc.stopProcessing();

	return 0;
}
