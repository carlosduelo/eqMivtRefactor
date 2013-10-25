/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlPlaneCache.h>

#include <lunchbox/sleep.h>
#include <lunchbox/clock.h>

#include <boost/lexical_cast.hpp>
#include <boost/progress.hpp>

eqMivt::hdf5File hdf5File;
eqMivt::ControlPlaneCache cpc;
float mO = 1.0f;


bool test(vmml::vector<3,int> start, vmml::vector<3,int> finish)
{
	if (!cpc.freeCacheAndPause() || !cpc.reSizeCacheAndContinue(start, finish))
	{
		std::cerr<<"Error, resizing plane cache"<<std::endl;
		return true;
	}

	int dim = (finish.y() - start.y())*(finish.z() - start.z());
	float * planeH = new float[dim];

	bool error = false;

	#ifndef DISK_TIMING 
		boost::progress_display show_progress(finish.x() - start.x());
	#endif

	for(int i = start.x(); i<finish.x() && !error; i++)
	{
		float * planeC = 0;

		do
		{
			planeC = cpc.getAndBlockPlane(i);
		}
		while(planeC == 0);

		hdf5File.readPlane(planeH, vmml::vector<3, int>(i, start.y(), start.z()), vmml::vector<3, int>(i, finish.y(), finish.z())); 

		for(int j=0; j<dim && !error; j++)
		{
			if (planeC[j] != planeH[j])
			{
				std::cerr<<"Error, planes differ "<<i<<" element "<<j<<" "<<planeC[j]<<" "<<planeH[j]<<std::endl;
				error = true;
			}
		}
		
		cpc.unlockPlane(i);	

		#ifndef DISK_TIMING 
			++show_progress;
		#endif
	}

	delete[] planeH;

	return error;
}

void testPerf(vmml::vector<3,int> start, vmml::vector<3,int> finish)
{
	if (!cpc.freeCacheAndPause() || !cpc.reSizeCacheAndContinue(start, finish))
	{
		std::cerr<<"Error, resizing plane cache"<<std::endl;
		return;
	}

	#ifndef DISK_TIMING 
		boost::progress_display show_progress(finish.x() - start.x());
	#endif

	for(int i = start.x(); i<finish.x(); i++)
	{
		float * planeC = 0;

		do
		{
			planeC = cpc.getAndBlockPlane(i);
		}
		while(planeC == 0);

		cpc.unlockPlane(i);	

		#ifndef DISK_TIMING 
			++show_progress;
		#endif
	}
}

int main(int argc, char ** argv)
{
	std::vector<std::string> parameters;
	parameters.push_back(std::string(argv[1]));
	parameters.push_back(std::string(argv[2]));
	
	if (argc == 4)
	{
		std::string n(argv[3]);
		mO = boost::lexical_cast<double>(n);
	}

	if (!cpc.initParameter(parameters, mO))
	{
		std::cerr<<"Error init control plane cache"<<std::endl;
		return 0;
	}

	hdf5File.init(parameters);

	vmml::vector<3, int> dim = hdf5File.getRealDimension();

	std::cout<<"Checking errors........."<<std::endl;

	bool error = false;
	for(int i=0; i<10 && !error; i++)
	{
		vmml::vector<3,int> s;
		vmml::vector<3,int> e;
		s.set(rand() % dim.x(), rand() % dim.y(), rand() % dim.z());
		do
		{
			e.set(rand() % (dim.x() - s.x()) + s.x(), rand() % (dim.y() - s.y()) + s.y(), rand() % (dim.z() - s.z()) + s.z());
		}
		while(s.x() >= e.x() || s.y() >= e.y() || s.z() >= e.z());

		std::cout<<"Test "<<i<<" "<<s<<" "<<e<<" : "<<std::endl;

		error = test(s, e);
		if (error)
			std::cout<<"Test Fail!"<<std::endl;
		else
		{
			std::cout<<"Test OK"<<std::endl;
		}
	}

	lunchbox::Clock clock;

	std::cout<<"Checking performance........."<<std::endl;

	for(int i=0; i<10 && !error; i++)
	{
		vmml::vector<3,int> s;
		vmml::vector<3,int> e;
		s.set(rand() % dim.x(), rand() % dim.y(), rand() % dim.z());
		do
		{
			e.set(rand() % (dim.x() - s.x()) + s.x(), rand() % (dim.y() - s.y()) + s.y(), rand() % (dim.z() - s.z()) + s.z());
		}
		while(s.x() >= e.x() || s.y() >= e.y() || s.z() >= e.z());

		double time = 0.0;
		clock.reset();
		testPerf(s, e);
		time = clock.getTimed()/1000.0;
		double bw = ((((e.x()-s.x())*(e.y()-s.y())*(e.z()-s.z()))*sizeof(float))/1204.0/1024.0)/time;

		std::cout<<"Test "<<s<<" "<<e<<": "<<time<<" seconds ~ "<<bw<<" MB/s"<<std::endl;
	}
	
	if (!error)
	{
		double time = 0.0;
		clock.reset();
		testPerf(vmml::vector<3,int>(0,0,0), dim);
		time = clock.getTimed()/1000.0;
		double bw = ((dim.x()*dim.y()*dim.z()*sizeof(float))/1204.0/1024.0)/time;

		std::cout<<"Read complete volume "<<dim<<" : "<<time<<" seconds ~ "<<bw<<" MB/s"<<std::endl; 
		if (test(vmml::vector<3,int>(0,0,0), dim))
		{
			std::cout<<"Test Fail!"<<std::endl;
		}
	}

	cpc.stopWork();

	hdf5File.close();

	return 0;
}
