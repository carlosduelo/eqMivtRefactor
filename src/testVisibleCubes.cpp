/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <visibleCubes.h>
#include <testVisibleCubes_CUDA.h>

#include <iostream>
#include <vector>

int main(int argc, char ** argv)
{

	eqMivt::VisibleCubes vC;

	vC.reSize(100);
	vC.init();

	std::vector<int> c = vC.getListCubes(CUBE);
	std::vector<int> nc = vC.getListCubes(NOCUBE);
	std::vector<int> ca = vC.getListCubes(CACHED);
	std::vector<int> nca = vC.getListCubes(NOCACHED);
	std::vector<int> p = vC.getListCubes(PAINTED);

	std::cout<<"CUBE size "<<c.size()<<std::endl;
	std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
	std::cout<<"CACHEDsize "<<ca.size()<<std::endl;
	std::cout<<"NOCACHED size "<<nca.size()<<std::endl;
	std::cout<<"PAINTED size "<<p.size()<<std::endl;

	std::vector<eqMivt::updateCube_t> changes;

	for(std::vector<int>::iterator it = nc.begin(); it!= nc.end(); ++it)
	{
		eqMivt::updateCube_t s;
		s.pixel = *it;
		s.cubeID = 21;

		if ((*it) % 4 == 0)
			s.state = NOCUBE;
		if ((*it) % 4 == 1)
			s.state = NOCACHED;
		if ((*it) % 4 == 2)
			s.state = CACHED;
		if ((*it) % 4 == 3)
			s.state = PAINTED;

		changes.insert(changes.end(), s);
	}

	vC.updateVisibleCubes(changes);

	c = vC.getListCubes(CUBE);
	nc = vC.getListCubes(NOCUBE);
	ca = vC.getListCubes(CACHED);
	nca = vC.getListCubes(NOCACHED);
	p = vC.getListCubes(PAINTED);

	std::cout<<"CUBE size "<<c.size()<<std::endl;
	std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
	std::cout<<"CACHEDsize "<<ca.size()<<std::endl;
	std::cout<<"NOCACHED size "<<nca.size()<<std::endl;
	std::cout<<"PAINTED size "<<p.size()<<std::endl;

	std::cout<<"________________________________________"<<std::endl;

	vC.updateGPU(CUBE , false, 0);
	std::cout<<"GPU size "<<vC.getSizeGPU()<<std::endl;

	test_updateCubesGPU(vC.getVisibleCubesGPU(), vC.getSizeGPU());

	vC.updateCPU();
	c = vC.getListCubes(CUBE);
	nc = vC.getListCubes(NOCUBE);
	ca = vC.getListCubes(CACHED);
	nca = vC.getListCubes(NOCACHED);
	p = vC.getListCubes(PAINTED);

	std::cout<<"CUBE size "<<c.size()<<std::endl;
	std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
	std::cout<<"CACHEDsize "<<ca.size()<<std::endl;
	std::cout<<"NOCACHED size "<<nca.size()<<std::endl;
	std::cout<<"PAINTED size "<<p.size()<<std::endl;

	vC.updateGPU(CUBE | NOCUBE, false, 0);
	std::cout<<"GPU size "<<vC.getSizeGPU()<<std::endl;

	test_updateCubesGPU(vC.getVisibleCubesGPU(), vC.getSizeGPU());

	vC.updateCPU();

	c = vC.getListCubes(CUBE);
	nc = vC.getListCubes(NOCUBE);
	ca = vC.getListCubes(CACHED);
	nca = vC.getListCubes(NOCACHED);
	p = vC.getListCubes(PAINTED);

	std::cout<<"CUBE size "<<c.size()<<std::endl;
	std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
	std::cout<<"CACHEDsize "<<ca.size()<<std::endl;
	std::cout<<"NOCACHED size "<<nca.size()<<std::endl;
	std::cout<<"PAINTED size "<<p.size()<<std::endl;


	vC.updateGPU(CUBE | NOCUBE | CACHED, false, 0);
	std::cout<<"GPU size "<<vC.getSizeGPU()<<std::endl;

	test_updateCubesGPU(vC.getVisibleCubesGPU(), vC.getSizeGPU());

	vC.updateCPU();

	c = vC.getListCubes(CUBE);
	nc = vC.getListCubes(NOCUBE);
	ca = vC.getListCubes(CACHED);
	nca = vC.getListCubes(NOCACHED);
	p = vC.getListCubes(PAINTED);

	std::cout<<"CUBE size "<<c.size()<<std::endl;
	std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
	std::cout<<"CACHEDsize "<<ca.size()<<std::endl;
	std::cout<<"NOCACHED size "<<nca.size()<<std::endl;
	std::cout<<"PAINTED size "<<p.size()<<std::endl;

	vC.updateGPU(CUBE | NOCUBE | CACHED | NOCACHED, false, 0);
	std::cout<<"GPU size "<<vC.getSizeGPU()<<std::endl;

	test_updateCubesGPU(vC.getVisibleCubesGPU(), vC.getSizeGPU());

	vC.updateCPU();

	c = vC.getListCubes(CUBE);
	nc = vC.getListCubes(NOCUBE);
	ca = vC.getListCubes(CACHED);
	nca = vC.getListCubes(NOCACHED);
	p = vC.getListCubes(PAINTED);

	std::cout<<"CUBE size "<<c.size()<<std::endl;
	std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
	std::cout<<"CACHEDsize "<<ca.size()<<std::endl;
	std::cout<<"NOCACHED size "<<nca.size()<<std::endl;
	std::cout<<"PAINTED size "<<p.size()<<std::endl;

	vC.updateGPU(CUBE | NOCUBE | CACHED | NOCACHED | PAINTED, false, 0);
	std::cout<<"GPU size "<<vC.getSizeGPU()<<std::endl;

	test_updateCubesGPU(vC.getVisibleCubesGPU(), vC.getSizeGPU());

	vC.updateCPU();

	c = vC.getListCubes(CUBE);
	nc = vC.getListCubes(NOCUBE);
	ca = vC.getListCubes(CACHED);
	nca = vC.getListCubes(NOCACHED);
	p = vC.getListCubes(PAINTED);

	std::cout<<"CUBE size "<<c.size()<<std::endl;
	std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
	std::cout<<"CACHEDsize "<<ca.size()<<std::endl;
	std::cout<<"NOCACHED size "<<nca.size()<<std::endl;
	std::cout<<"PAINTED size "<<p.size()<<std::endl;

	return 0;

}
