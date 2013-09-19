/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <visibleCubes.h>
#include <testVisibleCubes_CUDA.h>

#include <iostream>
#include <vector>

eqMivt::VisibleCubes vC;

int test(eqMivt::statusCube fromState, eqMivt::statusCube toState)
{
	std::vector<int> c = vC.getListCubes(CUBE);
	std::vector<int> nc = vC.getListCubes(NOCUBE);
	std::vector<int> ca = vC.getListCubes(CACHED);
	std::vector<int> p = vC.getListCubes(PAINTED);

	int ec = c.size();
	int enc = nc.size();
	int eca = ca.size();
	int ep = p.size();

	int add = 0;
	if ((fromState & CUBE) != NONE)
	{
		add += ec;
		ec = 0;
	}
	if ((fromState & NOCUBE) != NONE)
	{
		add += enc;
		enc = 0;
	}
	if ((fromState & CACHED) != NONE)
	{
		add += eca;
		eca = 0;
	}
	if ((fromState & PAINTED) != NONE)
	{
		add += ep;
		ep = 0;
	}
	switch(toState)
	{
		case CUBE:
			ec += add;
			break;
		case NOCUBE:
			enc += add;
			break;
		case CACHED:
			eca += add;
			break;
		case PAINTED:
			ep += add;
			break;
		#ifndef NDEBUG
		default:
			std::cerr<<"TEST Visible cubes, error toState worng"<<std::endl;
			throw;
		#endif
	}

	vC.updateGPU(fromState , false, 0);

	test_updateCubesGPU(vC.getVisibleCubesGPU(), vC.getSizeGPU(), toState);

	vC.updateCPU();
	c = vC.getListCubes(CUBE);
	nc = vC.getListCubes(NOCUBE);
	ca = vC.getListCubes(CACHED);
	p = vC.getListCubes(PAINTED);

	if (c.size() != ec		||
		nc.size() != enc	||
		ca.size() != eca	||
		p.size() != ep)
	{
		std::cout<<"CUBE size "<<c.size()<<" should be "<<ec<<std::endl;
		std::cout<<"NOCUBE size "<<nc.size()<<" should be "<<enc<<std::endl;
		std::cout<<"CACHED size "<<ca.size()<<" should be "<<eca<<std::endl;
		std::cout<<"PAINTED size "<<p.size()<<" should be "<<ep<<std::endl;
		std::cerr<<"Test error"<<std::endl;
		return false;
	}

	return true;
}

int main(int argc, char ** argv)
{
	int start = 25;
	int dim = 4*start;

	vC.reSize(dim);
	vC.init();

	std::vector<int> c = vC.getListCubes(CUBE);
	std::vector<int> nc = vC.getListCubes(NOCUBE);
	std::vector<int> ca = vC.getListCubes(CACHED);
	std::vector<int> p = vC.getListCubes(PAINTED);

	if (c.size() != 0		||
		nc.size() != 100	||
		ca.size() != 0		||
		p.size() != 0)
	{
		std::cout<<"CUBE size "<<c.size()<<std::endl;
		std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
		std::cout<<"CACHED size "<<ca.size()<<std::endl;
		std::cout<<"PAINTED size "<<p.size()<<std::endl;
		std::cerr<<"Test error"<<std::endl;
		return 0;
	}

	std::vector<eqMivt::visibleCube_t> changes;

	for(std::vector<int>::iterator it = nc.begin(); it!= nc.end(); ++it)
	{
		eqMivt::visibleCube_t s;
		s.pixel = *it;
		s.cubeID = 21;

		if ((*it) % 4 == 0)
			s.state = NOCUBE;
		if ((*it) % 4 == 1)
			s.state = CUBE;
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
	p = vC.getListCubes(PAINTED);

	if (nc.size() != start	||
		ca.size() != start	||
		c.size() != start	||
		p.size() != start)
	{
		std::cout<<"CUBE size "<<c.size()<<std::endl;
		std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
		std::cout<<"CACHED size "<<ca.size()<<std::endl;
		std::cout<<"PAINTED size "<<p.size()<<std::endl;
		std::cerr<<"Test error"<<std::endl;
		return 0;
	}

	bool  result =	test(CUBE, PAINTED) &&
					test(PAINTED, CUBE) &&
					test(CUBE, NOCUBE) &&
					test(CACHED, NOCUBE) &&
					test(NOCUBE | PAINTED, NOCUBE) &&
					test(NOCUBE, PAINTED) &&
					test(CUBE | NOCUBE | CACHED, PAINTED) &&
					test( PAINTED | NOCUBE, NOCUBE);

	if (result)
		std::cout<<" Test OK"<<std::endl;
	else
		std::cout<<" Test Fail"<<std::endl;

	return 0;

	
	#if 0
	vC.updateGPU(CUBE , false, 0);

	test_updateCubesGPU(vC.getVisibleCubesGPU(), vC.getSizeGPU(), PAINTED);

	vC.updateCPU();
	c = vC.getListCubes(CUBE);
	nc = vC.getListCubes(NOCUBE);
	ca = vC.getListCubes(CACHED);
	p = vC.getListCubes(PAINTED);

	if (c.size() != 0		||
		nc.size() != 25		||
		ca.size() != 25		||
		p.size() != 50)
	{
		std::cout<<"CUBE size "<<c.size()<<std::endl;
		std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
		std::cout<<"CACHED size "<<ca.size()<<std::endl;
		std::cout<<"PAINTED size "<<p.size()<<std::endl;
		std::cerr<<"Test error"<<std::endl;
		return 0;
	}

	vC.updateGPU(CUBE | NOCUBE, false, 0);

	test_updateCubesGPU(vC.getVisibleCubesGPU(), vC.getSizeGPU(), PAINTED);

	vC.updateCPU();

	c = vC.getListCubes(CUBE);
	nc = vC.getListCubes(NOCUBE);
	ca = vC.getListCubes(CACHED);
	p = vC.getListCubes(PAINTED);

	if (c.size() != 0		||
		nc.size() != 0		||
		ca.size() != 25		||
		p.size() != 75)
	{
		std::cout<<"CUBE size "<<c.size()<<std::endl;
		std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
		std::cout<<"CACHED size "<<ca.size()<<std::endl;
		std::cout<<"PAINTED size "<<p.size()<<std::endl;
		std::cerr<<"Test error"<<std::endl;
		return 0;
	}


	vC.updateGPU(CUBE | NOCUBE | CACHED, false, 0);

	test_updateCubesGPU(vC.getVisibleCubesGPU(), vC.getSizeGPU(), PAINTED);

	vC.updateCPU();

	c = vC.getListCubes(CUBE);
	nc = vC.getListCubes(NOCUBE);
	ca = vC.getListCubes(CACHED);
	p = vC.getListCubes(PAINTED);

	if (c.size() != 0		||
		nc.size() != 0		||
		ca.size() != 0		||
		p.size() != 100)
	{
		std::cout<<"CUBE size "<<c.size()<<std::endl;
		std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
		std::cout<<"CACHED size "<<ca.size()<<std::endl;
		std::cout<<"PAINTED size "<<p.size()<<std::endl;
		std::cerr<<"Test error"<<std::endl;
		return 0;
	}

	vC.updateGPU(CUBE | NOCUBE | CACHED | PAINTED, false, 0);

	test_updateCubesGPU(vC.getVisibleCubesGPU(), vC.getSizeGPU(), PAINTED);

	vC.updateCPU();

	c = vC.getListCubes(CUBE);
	nc = vC.getListCubes(NOCUBE);
	ca = vC.getListCubes(CACHED);
	p = vC.getListCubes(PAINTED);

	if (c.size() != 0		||
		nc.size() != 0		||
		ca.size() != 0		||
		p.size() != 100)
	{
		std::cout<<"CUBE size "<<c.size()<<std::endl;
		std::cout<<"NOCUBE size "<<nc.size()<<std::endl;
		std::cout<<"CACHED size "<<ca.size()<<std::endl;
		std::cout<<"PAINTED size "<<p.size()<<std::endl;
		std::cerr<<"Test error"<<std::endl;
		return 0;
	}
	#endif
}
