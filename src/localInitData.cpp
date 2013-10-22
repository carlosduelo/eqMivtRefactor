/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#include "localInitData.h"

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>

#include <vector>

namespace eqMivt
{
LocalInitData::LocalInitData()
        : _maxFrames( 0xffffffffu )
		, _isResident( false )
{
}

const LocalInitData& LocalInitData::operator = ( const LocalInitData& from )
{
    _maxFrames   = from._maxFrames;
    _isResident  = from._isResident;

    setOctreeFilename(from.getOctreeFilename());
	setDataFilename(from.getDataFilename());
	setMemoryOccupancy(from.getMemoryOccupancy());
	setTransferFunctionFile(from.getTransferFunctionFile());

    return *this;
}


bool LocalInitData::parseArguments( const int argc, char** argv )
{
    // Declare the supported options.
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
    ("version,v", "print version")
    ("help", "produce help message")
    ("eq-config", "Select equalizer configuration file")
    ("eq-layout", "Select equalizer layout in configuration file")
    ("octree-file,o", boost::program_options::value< std::vector<std::string> >()->multitoken(), "octree-file-path")
    ("data-file,d", boost::program_options::value< std::vector<std::string> >()->multitoken(), "hdf5-file dataset-name")
	("memory-occupancy,m", boost::program_options::value<float>(), "set memory occupancy value (0.0 , 1.0] optional")
    ("transfer-function,t", boost::program_options::value< std::vector<std::string> >()->multitoken(), "transfer function color path")
    ;

	boost::program_options::variables_map vm;
	try
	{
		boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
		boost::program_options::notify(vm);
	}
	catch( ... )
	{
        std::cerr << desc << std::endl;
		return false;
	}

    if (argc == 1 || vm.count("help"))
    {
        std::cout << desc << std::endl;
		return false;
    }
    if (vm.count("version"))
    {
        std::cout << "Version eqMivt: "<<VERSION_EQ_MIVT << std::endl;
		return false;
    }
	if (vm.count("memory-occupancy"))
	{
		float mO = vm["memory-occupancy"].as<float>();
		if (mO <= 0.0f || mO > 1.0f)
		{
			std::cerr<<"Memory occupancy may be > 0.0 and <= 1.0f"<<std::endl;
			std::cout << desc << std::endl;
			return false;
		}
		setMemoryOccupancy(mO);
	}
	else 
	{
		setMemoryOccupancy(1.0f);
	}

	bool printHelp = false;
    if (vm.count("octree-file"))
    {
        std::vector<std::string> octreefiles = vm["octree-file"].as< std::vector<std::string> >();

		if (octreefiles.size() != 1)
			printHelp = true;
		else
			setOctreeFilename(octreefiles[0]);
    }
    // Parameter needed
    else
    {
		setOctreeFilename("");
		printHelp = true;
    }

	if (vm.count("transfer-function"))
	{
		std::vector<std::string> files = vm["transfer-function"].as< std::vector<std::string> >();
		setTransferFunctionFile(files[0]);
	}
	else
	{
		setTransferFunctionFile("");
	}

	if (vm.count("data-file"))
	{
		std::vector<std::string> dataParam = vm["data-file"].as< std::vector<std::string> >();

		if (dataParam.size() != 2)
			printHelp = true;
		else
		{
			std::vector<std::string> fileParams;

			fileParams.push_back(dataParam[0]);
			fileParams.push_back(dataParam[1]);

			setDataFilename(fileParams);
		}
	}
    // Parameter needed
    else
    {
		std::vector<std::string> fileParams;
		setDataFilename(fileParams);
		printHelp = true;
    }

	if (printHelp)
        std::cerr << desc << std::endl;

    return !printHelp;
}
}

