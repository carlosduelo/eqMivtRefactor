/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <cuda_help.h>
#include <renderPNG.h>
#include <resourcesManager.h>

#include <vmmlib/matrix.hpp>

#include <lunchbox/clock.h>
#include <lunchbox/thread.h>
#include <lunchbox/condition.h>

eqMivt::ResourcesManager rM;
int device = 0;

class worker : public lunchbox::Thread
{
	private:
	lunchbox::Condition _cond;
	lunchbox::Condition _condE;

	vmml::vector<4, float> _up;
	vmml::vector<4, float> _right;
	vmml::vector<4, float> _origin;
	vmml::vector<4, float> _LB;
	float _w;
	float _h;

	eqMivt::RenderPNG	_render;

	bool _end;
	bool _endFrame;

	public:

	bool init(int device, std::string name)
	{
		if (!_render.init(device, name))
		{
			std::cerr<<"Error init render"<<std::endl;
			return false;
		}
		_render.setDrawCubes(false);
		return true;
	}

	bool setViewPort(int pvpW, int pvpH)
	{
		if (!_render.setViewPort(pvpW, pvpH))
		{
			std::cerr<<"Error setting viewport"<<std::endl;
			return false;
		}
		return true;
	}

	void end()
	{
		_cond.lock();
		_end = true;
		_cond.signal();
		_cond.unlock();
	}

	void startFrame(vmml::vector<4, float> up, vmml::vector<4, float> right, vmml::vector<4, float> origin, vmml::vector<4, float> LB, float w, float h)
	{
		_cond.lock();

		_up = up;
		_right = right;
		_origin = origin;
		_LB = LB;
		_w = w;
		_h = h;

		_cond.signal();
		_cond.unlock();
	}

	void endFrame()
	{
		_condE.lock();
		
		#if 1
		while(!_endFrame)
		{
			if (_condE.timedWait(1000))
				break;
			std::cerr<<"waiting frame"<<std::endl;
		}
		#else
			_condE.wait();
		#endif
		_condE.unlock();
	}

	virtual void run()
	{
		_end = false;

		std::cout<<"HOLA"<<std::endl;

		_cond.lock();
		while(1)
		{
			_cond.wait();
			if (_end)
				break;
			_condE.lock();
			_endFrame = false;
			if (!rM.updateRender(&_render))
			{
				std::cerr<<"Error updating render"<<std::endl;
				throw;
			}

			std::cout<<"RENDER"<<std::endl;
			if (!_render.frameDraw(_origin, _LB, _up, _right, _w, _h))
			{
				std::cerr<<"Error rendering"<<std::endl;
				throw;
			}
			
			//_render
			_endFrame = true;
			_condE.signal();
			_condE.unlock();
		}
		_cond.unlock();
	}
	
};


bool test2()
{
	int threads = 2;

	int pvpW = 1024;
	int pvpH = 1024;
	float tnear = 1.0f;
	float fov = 30;

	worker wo[threads];

	wo[0].start();
	wo[1].start();

	wo[0].init(device, "left");
	wo[1].init(device, "right");

	wo[0].setViewPort(pvpW, pvpH);
	wo[1].setViewPort(pvpW, pvpH);

	for(int f=0; f<rM.getNumOctrees(); f++)
	{
		vmml::vector<3, int> startV = rM.getStartCoord();
		vmml::vector<3, int> endV = rM.getEndCoord();
		vmml::vector<4, float> origin(startV.x() + ((endV.x()-startV.x())/3.0f), rM.getMaxHeight(), 1.1f*endV.z(), 1.0f);

		vmml::matrix<4,4,float> positionM = vmml::matrix<4,4,float>::IDENTITY;
		positionM.set_translation(vmml::vector<3,float>(origin.x(), origin.y(), origin.z()));
		vmml::matrix<4,4,float> model = vmml::matrix<4,4,float>::IDENTITY;
		model = positionM * model;

		vmml::vector<4, float> up(0.0f, 1.0f, 0.0f, 0.0f);
		vmml::vector<4, float> right(1.0f, 0.0f, 0.0f, 0.0f);
		float ft = tan(fov*M_PI/180);
		vmml::vector<4, float>LB(-ft, -ft, -tnear, 1.0f); 	
		vmml::vector<4, float>LT(-ft, ft, -tnear, 1.0f); 	
		vmml::vector<4, float>RT(0.0f, ft, -tnear, 1.0f); 	
		vmml::vector<4, float>RB(0.0f, -ft, -tnear, 1.0f); 	
		vmml::vector<4, float>LB2(0.0f, -ft, -tnear, 1.0f); 	
		vmml::vector<4, float>LT2(0.0f, ft, -tnear, 1.0f); 	
		vmml::vector<4, float>RT2(ft, ft, -tnear, 1.0f); 	
		vmml::vector<4, float>RB2(ft, -ft, -tnear, 1.0f); 	
		LB = model*LB;
		LT = model*LT;
		RB = model*RB;
		RT = model*RT;
		LB2 = model*LB2;
		LT2 = model*LT2;
		RB2 = model*RB2;
		RT2 = model*RT2;
		float w = (RB.x() - LB.x())/(float)pvpW;
		float h = (LT.y() - LB.y())/(float)pvpH;
		float w2 = (RB2.x() - LB2.x())/(float)pvpW;
		float h2 = (LT2.y() - LB2.y())/(float)pvpH;
		std::cout<<"Camera position "<<origin<<std::endl;
		std::cout<<"Frustum left"<<std::endl;
		std::cout<<LB<<std::endl;
		std::cout<<LT<<std::endl;
		std::cout<<RB<<std::endl;
		std::cout<<RT<<std::endl;
		std::cout<<"Frustum right"<<std::endl;
		std::cout<<LB2<<std::endl;
		std::cout<<LT2<<std::endl;
		std::cout<<RB2<<std::endl;
		std::cout<<RT2<<std::endl;

		up = LT-LB;
		right = RB - LB;
		right.normalize();
		up.normalize();

		
		wo[0].startFrame(up, right, origin, LB, w, h);
		wo[1].startFrame(up, right, origin, LB2, w2, h2);

		wo[0].endFrame();
		wo[1].endFrame();

		if (f < rM.getNumOctrees()-1 && !rM.loadNext())
		{
			std::cerr<<"Error loading next isosurface"<<std::endl;
			return false;
		}
	}

	wo[0].end();
	wo[1].end();

	wo[0].join();
	wo[1].join();

	return true;
}

int main(int argc, char ** argv)
{
	std::vector<std::string> parameters;
	parameters.push_back(std::string(argv[1]));
	parameters.push_back(std::string(argv[2]));

	device = eqMivt::getBestDevice();

	cudaFuncCache cacheConfig = cudaFuncCachePreferL1;
	if (cudaSuccess != cudaSetDevice(device) || cudaSuccess != cudaDeviceSetCacheConfig(cacheConfig))
	{
		std::cerr<<"Error setting up best device"<<std::endl;
		return 0;
	}

	if (argc == 5)
	{
		if (!rM.init(parameters, argv[3], argv[3]))
		{
			std::cerr<<"Error init resources manager"<<std::endl;
			return 0;
		}
	}
	else
	{
		if (!rM.init(parameters, argv[3], ""))
		{
			std::cerr<<"Error init resources manager"<<std::endl;
			return 0;
		}
	}
	if (!rM.start())
	{
		std::cerr<<"Error start resources manager"<<std::endl;
		return 0;
	}
	
	std::cout<<"============ Creating pictures ============"<<std::endl;

	if (test2())
	{
		std::cout<<"Test ok"<<std::endl;
	}
	else
	{
		std::cout<<"Test Fail"<<std::endl;
	}

	rM.destroy();

	std::cout<<"End test"<<std::endl;
}
