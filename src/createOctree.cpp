/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlCubeCache.h>

#include <octreeConstructor.h>
#include <createOctree_cuda.h>
#include <mortonCodeUtil_CPU.h>
#include <cuda_help.h>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <boost/progress.hpp>

#include <vector>
#include <queue>
#include <iostream>
#include <fstream>

#include <lunchbox/clock.h>
#include <lunchbox/thread.h>
#include <lunchbox/condition.h>
#include <lunchbox/thread.h>

#define MAX_DIMENSION 1024.0f
#define MIN_DIMENSION 128

struct octreeParameter_t
{
	vmml::vector<3, int>	start;
	vmml::vector<3, int>	end;
	std::vector<float>		isos;
	int						nLevels;
	int						maxLevel;
};

bool						disk;
std::vector<std::string>	file_params;
std::string					config_file_name;
std::string					octree_file_name;
std::vector<octreeParameter_t> octreeList;
int	numOctrees;
float memoryOccupancy;

std::vector<eqMivt::octreeConstructor *> octreesT;

vmml::vector<3, int> realDimVolume;
vmml::vector<3, int> dimVolume;

eqMivt::ControlCubeCPUCache cccCPU;
eqMivt::ControlCubeCache ccc;

int device = 0;

float toFloat(std::string  s)
{
	try
	{
		float r = -1.0f;
		r = boost::lexical_cast<float>(s);
		if (r < 0)
		{
			std::cerr<<"coordinates should be > 0"<<std::endl;
			return -1.0f;
		}
		return r;
	}
	catch( boost::bad_lexical_cast const& )
	{
		std::cerr <<"error parsing file" << std::endl;
		return  -1.0f;
	}

}

int toInt(std::string  s)
{
	try
	{
		int r = 0;
		r = boost::lexical_cast<int>(s);
		if (r < 0)
		{
			std::cerr<<"coordinates should be > 0"<<std::endl;
			return -1;
		}
		return r;
	}
	catch( boost::bad_lexical_cast const& )
	{
		std::cerr <<"error parsing file" << std::endl;
		return  -1;
	}

}

bool parseConfigFile(std::string file_name)
{
	std::ifstream infile;
	try
	{
		infile.open(file_name.c_str());
	}
	catch(...)
	{
		std::cerr<<file_name<<" do not exist"<<std::endl;
		return false;
	}

	if (!infile.is_open())
	{
		std::cerr<<file_name<<" do not exist"<<std::endl;
		return false;
	}

	std::string line;
	while (std::getline(infile, line))
	{
		boost::char_separator<char> sep(" ");
		boost::tokenizer< boost::char_separator<char> > tokens(line, sep);
		
		int size = 0;
		int isosPos = 0;
		bool findL = false;
		for(boost::tokenizer< boost::char_separator<char> >::iterator tok_iter= tokens.begin(); tok_iter!=tokens.end(); tok_iter++)
		{
			if (!findL)
			{
				if ("l" == *tok_iter)
					findL = true;
				else
					isosPos++;
			}
			size++;
		}

		if (size < 3 || !findL || (isosPos != 2 && isosPos != 6))
		{
			std::cerr<<"Error parsing config file"<<std::endl;
			std::cerr<<"X0 [Y0 Z0] X1 [ Y1 Z1 ] l iso1 iso2 iso3 ..."<<std::endl;
			std::cerr<<"Please, if you provide Y0 and Z0, provide Y1 and Z1 as well."<<std::endl;
			std::cerr<<line<<std::endl;
			infile.close();
			return false;
		}
		
		octreeParameter_t param;
		boost::tokenizer< boost::char_separator<char> >::iterator tok_iter = tokens.begin();

		param.start[0] = toInt(*tok_iter);	if (param.start[0] < 0) {infile.close(); return false;} tok_iter++;

		if (isosPos == 2)
		{
			param.start[1] = 0;
			param.start[2] = 0;
		}
		else
		{
			param.start[1] = toInt(*tok_iter);	if (param.start[1] < 0) {infile.close(); return false;} tok_iter++;
			param.start[2] = toInt(*tok_iter);	if (param.start[2] < 0) {infile.close(); return false;} tok_iter++;
		}

		if (param.start[0] >= realDimVolume[0] || param.start[1] >= realDimVolume[1] ||param.start[2] >= realDimVolume[2])
		{
			std::cerr<<"Error parsing config file"<<std::endl;
			std::cerr<<"X0 [Y0 Z0] X1 [ Y1 Z1 ] l iso1 iso2 iso3 ..."<<std::endl;
			std::cerr<<"Start coordinate is not inside the volume"<<std::endl;
			std::cerr<<line<<std::endl;
			infile.close();
			return false;

		}

		param.end[0] = toInt(*tok_iter); if (param.end[0] < 0) {infile.close(); return false; } tok_iter++;
		
		if (isosPos == 2)
		{
			param.end[1] = realDimVolume[1]; 
			param.end[2] = realDimVolume[2];
		}
		else
		{
			if ("M" == *tok_iter)
			{
				param.end[1] = realDimVolume[1];
			}
			else
			{
				param.end[1] = toInt(*tok_iter); if (param.end[1] < 0) {infile.close(); return false; }
			}
			tok_iter++;
			if ("M" == *tok_iter)
			{
				param.end[2] = realDimVolume[2];
			}
			else
			{
				param.end[2] = toInt(*tok_iter); if (param.end[2] < 0) {infile.close(); return false; }
			}
			tok_iter++;
		}

		if (param.end[0] > realDimVolume[0] || param.end[1] > realDimVolume[1] || param.end[2] > realDimVolume[2] ||
			param.start[0] >= param.end[0] || param.start[1] >= param.end[1] || param.start[2] >= param.end[2])
		{
			std::cerr<<"Error parsing config file"<<std::endl;
			std::cerr<<"X0 [Y0 Z0] X1 [ Y1 Z1 ] l iso1 iso2 iso3 ..."<<std::endl;
			std::cerr<<"Start coordinate is not inside the volume"<<std::endl;
			std::cerr<<line<<std::endl;
			infile.close();
			return false;

		}

		if ("l" != *tok_iter)
		{
			std::cerr<<"Error parsing config file"<<std::endl;
			std::cerr<<"X0 [Y0 Z0] X1 [ Y1 Z1 ] l iso1 iso2 iso3 ..."<<std::endl;
			std::cerr<<"Please, if you provide Y0 and Z0, provide Y1 and Z1 as well."<<std::endl;
			std::cerr<<line<<std::endl;
			infile.close();
			return false;
		}
		tok_iter++;

		while(tok_iter != tokens.end())
		{
			float i = toFloat(*tok_iter);
			if (i <= 0.0f)
			{
				std::cerr<<"Isosurface should be > 0.0, "<<i<<std::endl;
				return false;
			}
			param.isos.push_back(i);
			tok_iter++;
			numOctrees++;
		}

		int dim = fmaxf(param.end[0] - param.start[0], fmaxf(param.end[1] - param.start[1], param.end[2] - param.start[2]));

		float aux = logf(dim)/logf(2.0f);
		float aux2 = aux - floorf(aux);
		param.nLevels = aux2>0.0 ? aux+1 : aux;

		float sizeD = 0.0f;
		param.maxLevel = 0;
		while(param.maxLevel != param.nLevels)
		{
			sizeD =        param.end[0] > dimVolume[0] ? (dimVolume[0] - param.start[0])/exp2f(param.nLevels - param.maxLevel) : (param.end[0] - param.start[0])/exp2f(param.nLevels - param.maxLevel);
			sizeD *= param.end[1] > dimVolume[1] ? (dimVolume[1] - param.start[1])/exp2f(param.nLevels - param.maxLevel) : (param.end[1] - param.start[1])/exp2f(param.nLevels - param.maxLevel);
			sizeD *= param.end[2] > dimVolume[2] ? (dimVolume[2] - param.start[2])/exp2f(param.nLevels - param.maxLevel) : (param.end[2] - param.start[2])/exp2f(param.nLevels - param.maxLevel);
			sizeD *= 0.5f; // Isosurface could be the half of the volume
			sizeD *= sizeof(eqMivt::index_node_t);
			sizeD /= 1024.0f;
			sizeD /= 1024.0f;

			if (sizeD <= MAX_DIMENSION)
				param.maxLevel++;
			else
				break;
		}


		if (param.nLevels < MIN_LEVEL)
		{
			std::cerr<<"Volume size should be at least greater than 32 yours is: "<<exp2(param.nLevels)<<std::endl;
			return false;
		}

		#ifndef NDEBUG
		std::cout<<"Start "<<param.start<<std::endl;
		std::cout<<"Finish "<<param.end<<std::endl;
		std::cout<<"nLevels "<<param.nLevels<<std::endl;
		std::cout<<"max Level "<<param.maxLevel<<std::endl;
		for(std::vector<float>::iterator it = param.isos.begin(); it!=param.isos.end(); it++)
			std::cout<<*it<<" ";
		std::cout<<std::endl;
		#endif

		octreeList.push_back(param);

	}

	infile.close();

	return true;
}

bool checkParameters(const int argc, char ** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message")
    ("data-file,d", boost::program_options::value< std::vector<std::string> >()->multitoken(), "hdf5_file-path data-set-name [x_grid y_grid z_grid]")
	("output-file-name,o", boost::program_options::value< std::vector<std::string> >()->multitoken(), "set name of output file, optional, by default same name as data with extension octree")
	("memory-occupancy,m", boost::program_options::value<float>(), "set memory occupancy value (0.0 , 1.0] optional")
	("config-file,f", boost::program_options::value< std::vector<std::string> >()->multitoken(), "config file")
	("use-disk,k", "use it when a limited memory machine")
    ;

	boost::program_options::variables_map vm;
	try
	{
		boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
		boost::program_options::notify(vm);
	}
	catch( ... )
	{
		std::cout<<"Octree constructor: allows create a octree"<<std::endl; 
        std::cout << desc << "\n";
		return false;
	}

    if (argc == 1 || vm.count("help"))
    {
        std::cout << desc << "\n";
		return false;
    }
    if (vm.count("use-disk"))
		disk = true;
	else
		disk = false;
	if (vm.count("memory-occupancy"))
	{
		float mO = vm["memory-occupancy"].as<float>();
		if (mO <= 0.0f || mO > 1.0f)
		{
			std::cerr<<"Memory occupancy may be > 0.0 and <= 1.0f"<<std::endl;
			return false;
		}
		 memoryOccupancy = mO;
	}
	else 
	{
		memoryOccupancy = 1.0f;
	}

	if (vm.count("data-file"))
	{
		std::vector<std::string> dataParam = vm["data-file"].as< std::vector<std::string> >();

		if (dataParam.size() != 2 && dataParam.size() != 5)
		{
			std::cerr <<"data-file option: file-path<string> dataset-name<string> [ xgrid ygrind zgrid]" << std::endl;
			return false;
		}
		
		file_params = dataParam;

		eqMivt::hdf5File h5File;
		if (!h5File.init(file_params))
		{
			std::cerr <<"data-file option: file-path<string> dataset-name<string> [ xgrid ygrind zgrid]" << std::endl;
			h5File.close();
			return false;
		}
		realDimVolume = h5File.getRealDimension();
		h5File.close();
		for(int i=0; i<3; i++)
		{
			float aux = logf(realDimVolume[i])/logf(2.0f);
			float aux2 = aux - floorf(aux);
			int n = aux2>0.0 ? aux+1 : aux;
			dimVolume[i] = exp2f(n);
		}


		if (vm.count("output-file-name"))
		{
			std::vector<std::string> dataParam = vm["output-file-name"].as< std::vector<std::string> >();

			octree_file_name = dataParam[0];
		}
		else
		{
			octree_file_name = file_params[0];
			octree_file_name.erase(octree_file_name.find_last_of("."), std::string::npos);
			octree_file_name += ".octree";
		}

		if (vm.count("config-file"))
		{
			std::vector<std::string> dataParam = vm["config-file"].as< std::vector<std::string> >();

			config_file_name = dataParam[0];
		}
		else
		{
			std::cout<<desc<<std::endl;
			return false;
		}

	}
	else
	{
		std::cout << desc << "\n";
		return false;
	}

	if (parseConfigFile(config_file_name))
	{
		std::cout<<"Parsing config file..... OK"<<std::endl;
		return true;
	}
	else
	{
		return false;
	}
}

class Worker : public lunchbox::Thread
{
	private:
	eqMivt::octreeConstructor * _o;
	unsigned int _lCPU;
	unsigned int _l;
	cudaStream_t stream;
	bool _showp;

	public:
	void setParameters(eqMivt::octreeConstructor * o, unsigned int lCPU, unsigned int l, bool showp)
	{
		_o = o;
		_lCPU = lCPU;
		_l = l;
		_showp = showp;
		if (cudaSuccess != cudaStreamCreate(&stream))
		{
			std::cerr<<"Error creating stream "<<device<<" : "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}
	}

	virtual void run()
	{
		boost::progress_display * showP = 0;

		if (cudaSuccess != cudaSetDevice(device))
		{
			std::cerr<<"Error setting device "<<device<<" : "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}
		
		int dimV = exp2(_o->getnLevels());
		int dimC = pow(exp2(_o->getnLevels() - _l), 3); 

		eqMivt::index_node_t idST = eqMivt::coordinateToIndex(vmml::vector<3,int>(0,0,0), _lCPU, _o->getnLevels());
		eqMivt::index_node_t idET = eqMivt::coordinateToIndex(vmml::vector<3,int>(dimV-1,dimV-1,dimV-1), _lCPU, _o->getnLevels());

		#ifndef DISK_TIMING 
			if (_showp)
				showP = new boost::progress_display(idET - idST + 1);
		#endif

		unsigned char * resultGPU = 0;
		unsigned char * resultCPU = new unsigned char[dimC];

		if (cudaSuccess != cudaMalloc((void**)&resultGPU, dimC*sizeof(unsigned char)))
		{
			std::cerr<<"Error allocating memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}

		for(eqMivt::index_node_t idT = idST; idT<=idET; idT++)
		{
			eqMivt::index_node_t idS = idT << (3*(_l - _lCPU)); 
			eqMivt::index_node_t idE = (idT+1) << (3*(_l - _lCPU));

			for(eqMivt::index_node_t id = idS; id<idE; id++)
			{
				if (ccc.checkCubeInside(id))
				{
					float * cube = 0;

					do
					{
						cube = ccc.getAndBlockElement(id);
					}
					while(cube == 0);

					if (cudaSuccess != cudaMemsetAsync((void*)resultGPU, 0, dimC*sizeof(unsigned char), stream))
					{
						std::cerr<<"Error init memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
						throw;
					}

					eqMivt::extracIsosurface(dimC, _l, _o->getnLevels(), _o->getIso(), id, resultGPU, cube, stream);

					if (cudaSuccess != cudaMemcpyAsync((void*)resultCPU, (void*)resultGPU, dimC*sizeof(unsigned char), cudaMemcpyDeviceToHost, stream))
					{
						std::cerr<<"Error copying memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
						throw;
					}

					if (cudaSuccess !=	cudaStreamSynchronize(stream))
					{
						std::cerr<<"Error sync stream "<<device<<" : "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
						throw;
					}

					eqMivt::index_node_t ids = id << (3*(_o->getMaxLevel() - _l));
					eqMivt::index_node_t ide = (id+1) << (3*(_o->getMaxLevel() - _l));
					int dim = pow(exp2(_o->getnLevels() - _o->getMaxLevel()), 3);
					int index = 0;

					for(eqMivt::index_node_t i=ids; i<ide; i++)
					{
						for(int j=0; j<dim; j++)
						{
							if (resultCPU[index + j] == (unsigned char) 1)
							{
								_o->addVoxel(i);
								break;
							}
						}
						index +=dim;
					}
					

					ccc.unlockElement(id);
				}
			}
		#ifndef DISK_TIMING 
			if (_showp)
				++(*showP);
		#endif
		}

		if (cudaSuccess != cudaFree((void*)resultGPU))
		{
			std::cerr<<"Error free memory: "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}
		if (cudaSuccess != cudaStreamDestroy(stream))
		{
			std::cerr<<"Error destroy stream "<<device<<" : "<<cudaGetErrorString(cudaGetLastError())<<std::endl;
			throw;
		}
	}

};

bool createOctree(octreeParameter_t p)
{
	std::vector<eqMivt::octreeConstructor *> oc;
	for(std::vector<float>::iterator it = p.isos.begin(); it!=p.isos.end(); it++)
	{
		eqMivt::octreeConstructor * o = new eqMivt::octreeConstructor(p.nLevels, p.maxLevel, *it, p.start, p.end, disk);
		oc.push_back(o);
		octreesT.push_back(o);
	}

	int dimV = exp2(p.nLevels); 
	vmml::vector<3, int> sP, eP;
	int cubeLevel = p.nLevels <= MIN_LEVEL ? 0 : p.nLevels - p.maxLevel > MIN_LEVEL ? p.maxLevel : p.nLevels - MIN_LEVEL;
	int levelCubeCPU = p.nLevels < 9 ? 0 : p.nLevels - 9;

	sP[0] = p.start.x() - CUBE_INC < 0 ? 0 : p.start.x() - CUBE_INC; 
	sP[1] = p.start.y() - CUBE_INC < 0 ? 0 : p.start.y() - CUBE_INC; 
	sP[2] = p.start.z() - CUBE_INC < 0 ? 0 : p.start.z() - CUBE_INC; 
	eP[0] = p.end.x() + CUBE_INC >= realDimVolume.x() ? realDimVolume.x() : p.end.x() + CUBE_INC;
	eP[1] = p.end.y() + CUBE_INC >= realDimVolume.y() ? realDimVolume.y() : p.end.y() + CUBE_INC;
	eP[2] = p.end.z() + CUBE_INC >= realDimVolume.z() ? realDimVolume.z() : p.end.z() + CUBE_INC;

	#ifndef NDEBUG
	std::cout<<"ReSize Plane Cache "<<sP<<" "<<eP<<std::endl;
	std::cout<<"Subset volume "<<p.start - vmml::vector<3,int>(CUBE_INC,CUBE_INC,CUBE_INC)<<" "<<p.start+vmml::vector<3,int>(dimV+CUBE_INC, dimV+CUBE_INC,dimV+CUBE_INC)<<std::endl;
	std::cout<<"ReSize Cube Cache nLevels "<<p.nLevels<<" level cube cpu "<<levelCubeCPU<<" level cube "<<cubeLevel<<" offset "<<p.start<<std::endl;
	#endif

	if (!cccCPU.freeCacheAndPause() || !cccCPU.reSizeCacheAndContinue(p.start, eP, levelCubeCPU, p.nLevels))
	{
		std::cerr<<"Error, resizing plane cache"<<std::endl;
		return false;
	}
	if (!ccc.freeCacheAndPause() || !ccc.reSizeCacheAndContinue(p.nLevels, cubeLevel, p.start))
	{
		std::cerr<<"Error, resizing plane cache"<<std::endl;
		return false;
	}

	Worker workers[p.isos.size()];

	workers[0].setParameters(oc[0], levelCubeCPU, cubeLevel, true);
	workers[0].start();
	for(unsigned int i=1; i<p.isos.size(); i++)
	{
		workers[i].setParameters(oc[i], levelCubeCPU, cubeLevel, false);
		workers[i].start();
	}

	for(unsigned int i=0; i<p.isos.size(); i++)
		workers[i].join();

	return true;
}

int main( const int argc, char ** argv)
{
	numOctrees = 0;

	if (!checkParameters(argc, argv))
		return 0;

	if (!cccCPU.initParameter(file_params, memoryOccupancy))
	{
		std::cerr<<"Error init control plane cache"<<std::endl;
		return 0;
	}

	device = eqMivt::getBestDevice();

	if (!ccc.initParameter(&cccCPU, device))
	{
		std::cerr<<"Error init control cube cache"<<std::endl;
	}

	lunchbox::Clock clock;

	for(std::vector<octreeParameter_t>::iterator it = octreeList.begin(); it!=octreeList.end(); it++)
	{
		clock.reset();
		createOctree(*it);
		std::cout<<"Time to create octree: "<<clock.getTimed()/1000.0<<" seconds."<<std::endl;
	}

	ccc.stopCache();
	cccCPU.stopCache();

	std::cout<<"Writing output file"<<std::endl;

	clock.reset();

	std::ofstream file(octree_file_name.c_str(), std::ofstream::binary);
	
	int magicWord = 919278872;

	eqMivt::hdf5File h5File;
	if (!h5File.init(file_params))
	{
		std::cerr <<"data-file option: file-path<string> dataset-name<string> [ xgrid ygrind zgrid]" << std::endl;
		h5File.close();
		return false;
	}
	float * xGrid = 0;
	float * yGrid = 0;
	float * zGrid = 0;
	if(	!h5File.getxGrid(&xGrid) ||
		!h5File.getyGrid(&yGrid) ||
		!h5File.getzGrid(&zGrid))
	{
		std::cerr<<"Error reading grids from hdf5 file"<<std::endl;
		return 0;
	}
	h5File.close();

	file.write((char*)&magicWord,  sizeof(magicWord));
	file.write((char*)&numOctrees,sizeof(numOctrees));
	file.write((char*)&realDimVolume.array,3*sizeof(int));
	file.write((char*)xGrid, realDimVolume[0]*sizeof(float));
	file.write((char*)yGrid, realDimVolume[1]*sizeof(float));
	file.write((char*)zGrid, realDimVolume[2]*sizeof(float));

	for(std::vector<octreeParameter_t>::iterator it=octreeList.begin(); it!=octreeList.end(); it++)
	{
		int nO = it->isos.size();
		file.write((char*)&nO, sizeof(int));
		file.write((char*)&it->start.array,3*sizeof(int));
		file.write((char*)&it->end.array,3*sizeof(int));
		file.write((char*)&it->nLevels, sizeof(int));
		file.write((char*)&it->maxLevel, sizeof(int));
	}

	for(std::vector<octreeParameter_t>::iterator it=octreeList.begin(); it!=octreeList.end(); it++)
		for(std::vector<float>::iterator itS = it->isos.begin(); itS!=it->isos.end(); itS++)
		{
			float iso = *itS;
			file.write((char*)&iso,sizeof(float));
		}
	
		// offset from start
		int initDesp =	5*sizeof(int) + realDimVolume[0]*sizeof(float) + realDimVolume[1]*sizeof(float) + realDimVolume[2]*sizeof(float) +
					octreeList.size() * 9 *sizeof(int) + numOctrees * sizeof(float);

		int desp	= 5*sizeof(int) + realDimVolume[0]*sizeof(float) + realDimVolume[1]*sizeof(float) + realDimVolume[2]*sizeof(float) +
						octreeList.size() * 9 *sizeof(int) + numOctrees * sizeof(float) + numOctrees*sizeof(int);

		int offsets[numOctrees];
		offsets[0] = desp;

		int i = 0;
		for(std::vector<eqMivt::octreeConstructor*>::iterator it=octreesT.begin(); it!=octreesT.end(); it++)
		{
			file.seekp(initDesp, std::ios_base::beg);
			file.write((char*)&desp, sizeof(desp));
			initDesp += sizeof(int);

			eqMivt::octreeConstructor * o = *it;
			o->completeOctree();	
			desp = o->getSize();
			if (i < numOctrees-1)
				offsets[i+1] = desp;

			file.seekp(offsets[0], std::ios_base::beg);
			for(int d=1; d<=i; d++)
				file.seekp(offsets[d], std::ios_base::cur);

			o->writeToFile(&file);
			i++;
		}

	file.close();

	std::cout<<"Time to write file "<<clock.getTimed()/1000.0<<" seconds."<<std::endl;

	for(std::vector<eqMivt::octreeConstructor*>::iterator it=octreesT.begin(); it!=octreesT.end(); it++)
	{
		eqMivt::octreeConstructor * o = *it;
		delete o;
	}

	return 0;
}
