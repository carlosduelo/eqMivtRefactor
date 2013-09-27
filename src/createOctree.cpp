/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include <controlCubeCache.h>

#include <octreeConstructor.h>
#include <createOctree_cuda.h>
#include <mortonCodeUtil_CPU.h>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <boost/progress.hpp>

#include <vector>
#include <iostream>
#include <fstream>

#define MAX_DIMENSION 512.0f
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

std::vector<eqMivt::octreeConstructor *> octreesT;

vmml::vector<3, int> realDimVolume;
vmml::vector<3, int> dimVolume;

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

		if (size < 4 || (isosPos != 4 && isosPos != 2))
		{
			std::cerr<<"Error parsing config file"<<std::endl;
			std::cerr<<"X0 [Y0 Z0] dim l iso1 iso2 iso3 ..."<<std::endl;
			std::cerr<<"dim has to be power of 2 and >= 128"<<std::endl;
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
			std::cerr<<"Start coordinates X0 Y0 Z0 has to be < volume dimension: "<<param.start<<" "<<realDimVolume<<std::endl;
			infile.close();
			return false;

		}

		int dim = toInt(*tok_iter);    if (dim <= 0) {infile.close(); return false;} tok_iter++; 

		// Dim power of 2 ?
		if (((dim & (dim - 1)) != 0) || dim < 128)
		{
			std::cerr<<"Error parsing config file"<<std::endl;
			std::cerr<<"X0 [Y0 Z0] dim l iso1 iso2 iso3 ..."<<std::endl;
			std::cerr<<"dim has to be power of 2 and >= 128"<<std::endl;
			infile.close();
			return false;
		}

		if ("l" != *tok_iter)
		{
			std::cerr<<"Error parsing config file"<<std::endl;
			std::cerr<<"X0 [Y0 Z0] dim l iso1 iso2 iso3 ..."<<std::endl;
			std::cerr<<"dim has to be power of 2 and >= 128"<<std::endl;
			infile.close();
			return false;
		}
		tok_iter++;

		while(tok_iter != tokens.end())
		{
			float i = toFloat(*tok_iter);
			if (i <= 0.0f)
			{
				std::cerr<<"Isosurface should be > 0.0"<<std::endl;
				return false;
			}
			param.isos.push_back(i);
			tok_iter++;
			numOctrees++;
		}

		float aux = logf(dim)/logf(2.0f);
		float aux2 = aux - floorf(aux);
		param.nLevels = aux2>0.0 ? aux+1 : aux;

		param.end[0] = param.start[0] + dim >= realDimVolume.x() ? realDimVolume.x() : param.start[0] + dim;
		param.end[1] = param.start[1] + dim >= realDimVolume.y() ? realDimVolume.y() : param.start[1] + dim;
		param.end[2] = param.start[2] + dim >= realDimVolume.z() ? realDimVolume.z() : param.start[2] + dim;

		float sizeD = 0.0f;
		param.maxLevel = 0;
		while(param.maxLevel != param.nLevels)
		{
			sizeD =	param.end[0] > dimVolume[0] ? (dimVolume[0] - param.start[0])/exp2f(param.nLevels - param.maxLevel) : (param.end[0] - param.start[0])/exp2f(param.nLevels - param.maxLevel);
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

void _checkCube_cuda(std::vector<eqMivt::octreeConstructor *> octrees, std::vector<float> isos, int nLevels, int nodeLevel, int dimNode, eqMivt::index_node_t idCube, int cubeLevel, int cubeDim, float * cube)
{
	vmml::vector<3, int> coorCubeStart = eqMivt::getMinBoxIndex2(idCube, cubeLevel, nLevels);
	vmml::vector<3, int> coorCubeFinish = coorCubeStart + (cubeDim - 2*CUBE_INC);
	int coorCubeStartV[3] = {coorCubeStart.x() - CUBE_INC, coorCubeStart.y() - CUBE_INC, coorCubeStart.z() - CUBE_INC};

	eqMivt::index_node_t start = idCube << 3*(nodeLevel - cubeLevel); 
	eqMivt::index_node_t finish = eqMivt::coordinateToIndex(coorCubeFinish - 1, nodeLevel, nLevels);

	double freeMemory = octreeConstructorGetFreeMemory();
	int inc = finish - start + 1;
	int rest = inc / 10;

	while(inc*sizeof(eqMivt::index_node_t) > freeMemory)
	{
		inc = inc - rest;
	}

	eqMivt::index_node_t * resultCPU = new eqMivt::index_node_t[inc];
	eqMivt::index_node_t * resultGPU = 0;
	resultGPU = octreeConstructorCreateResult(inc);
	if (resultGPU == 0)
	{
		std::cerr<<"Error creating a structure in a cuda device"<<std::endl;
		throw;
	}

	for(eqMivt::index_node_t id=start; id<=finish; id+=inc)
	{
		for(int i=0; i<octrees.size(); i++)
		{
			int num = (id + inc) > finish ? finish - id  + 1: inc;
			octreeConstructorComputeCube(resultGPU, num, id, isos[i], cube, nodeLevel, nLevels, dimNode, cubeDim, coorCubeStartV);

			bzero(resultCPU, inc*sizeof(eqMivt::index_node_t));

			if (!octreeConstructorCopyResult(resultCPU, resultGPU, num))
			{
				std::cerr<<"Error copying structures from a cuda device"<<std::endl;
				throw;
			}
			for(int j=0; j<num; j++)
			{
				if (resultCPU[j] != (eqMivt::index_node_t)0)
				{
					#if 0
					vmml::vector<3, int> coorNodeStart = eqMivt::getMinBoxIndex2(resultCPU[j], nodeLevel, nLevels);
					vmml::vector<3, int> coorNodeFinish = coorNodeStart + dimNode - 1;
					#endif
					octrees[i]->addVoxel(resultCPU[j]);
				}
			}
		}
	}

	delete[] resultCPU;
	octreeConstructorDestroyResult(resultGPU);
}

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
	int dimNode = exp2(p.nLevels - p.maxLevel);
	int cubeLevel = 0;
	int cubeDim = dimV;

	while(cubeDim > 64)
	{
		cubeLevel++;
		cubeDim = exp2(p.nLevels - cubeLevel);
	}

	sP[0] = p.start.x() - CUBE_INC < 0 ? 0 : p.start.x() - CUBE_INC; 
	sP[1] = p.start.y() - CUBE_INC < 0 ? 0 : p.start.y() - CUBE_INC; 
	sP[2] = p.start.z() - CUBE_INC < 0 ? 0 : p.start.z() - CUBE_INC; 
	eP[0] = p.start.x() + dimV + CUBE_INC >= realDimVolume.x() ? realDimVolume.x() : p.start.x() + dimV + CUBE_INC;
	eP[1] = p.start.y() + dimV + CUBE_INC >= realDimVolume.y() ? realDimVolume.y() : p.start.y() + dimV + CUBE_INC;
	eP[2] = p.start.z() + dimV + CUBE_INC >= realDimVolume.z() ? realDimVolume.z() : p.start.z() + dimV + CUBE_INC;

	#ifndef NDEBUG
	std::cout<<"ReSize Plane Cache "<<sP<<" "<<eP<<std::endl;
	std::cout<<"Subset volume "<<p.start - vmml::vector<3,int>(CUBE_INC,CUBE_INC,CUBE_INC)<<" "<<p.start+vmml::vector<3,int>(dimV+CUBE_INC, dimV+CUBE_INC,dimV+CUBE_INC)<<std::endl;
	std::cout<<"ReSize Cube Cache nLevels "<<p.nLevels<<" level cube "<<cubeLevel<<" offset "<<p.start<<std::endl;
	#endif

	eqMivt::ControlPlaneCache cpc;
	cpc.initParameter(file_params);
	cpc.start();

	eqMivt::ControlCubeCache ccc;
	ccc.initParameter(&cpc);
	ccc.start();

	cpc.reSize(sP, eP);
	ccc.reSize(p.nLevels, cubeLevel, p.start);

	float * cube = 0;

	eqMivt::index_node_t idS = eqMivt::coordinateToIndex(vmml::vector<3,int>(0,0,0), cubeLevel, p.nLevels);
	eqMivt::index_node_t idE = eqMivt::coordinateToIndex(vmml::vector<3,int>(dimV-1,dimV-1,dimV-1), cubeLevel, p.nLevels);

	#ifndef DISK_TIMING 
		boost::progress_display show_progress(idE-idS+1);
	#endif

	for(eqMivt::index_node_t id = idS; id<=idE; id++)
	{
		vmml::vector<3,int> c = eqMivt::getMinBoxIndex2(id, ccc.getCubeLevel(), p.nLevels) + p.start;
		if (c.x() < realDimVolume.x() || c.y() < realDimVolume.y() || c.z() < realDimVolume.z())
		{
			cube = 0;
			do
			{
				cube = ccc.getAndBlockCube(id);
			}
			while(cube == 0);

			_checkCube_cuda(oc, p.isos, p.nLevels, p.maxLevel, dimNode, id, ccc.getCubeLevel(), ccc.getDimCube(), cube);
		}
		ccc.unlockCube(id);
		#ifndef DISK_TIMING 
			++show_progress;
		#endif
	}

	ccc.stopProcessing();
	cpc.stopProcessing();
}

int main( const int argc, char ** argv)
{
	numOctrees = 0;

	if (!checkParameters(argc, argv))
		return 0;


	for(std::vector<octreeParameter_t>::iterator it = octreeList.begin(); it!=octreeList.end(); it++)
	{
		createOctree(*it);
	}

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

	for(std::vector<eqMivt::octreeConstructor*>::iterator it=octreesT.begin(); it!=octreesT.end(); it++)
	{
		eqMivt::octreeConstructor * o = *it;
		delete o;
	}

	return 0;
}
