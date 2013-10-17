/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <hdf5File.h>

#include <mortonCodeUtil_CPU.h>

#include <iostream>
#include <strings.h>

#ifdef DISK_TIMING
#include <lunchbox/clock.h>
#endif

namespace eqMivt
{

bool hdf5File::init(std::vector<std::string> file_params) 
{
	if ((_file_id    = H5Fopen(file_params[0].c_str(), H5F_ACC_RDWR, H5P_DEFAULT)) < 0)
	{
		std::cerr<<"hdf5: opening "<<file_params[0]<<std::endl;
		return false;
	}

	if ((_dataset_id = H5Dopen1(_file_id, file_params[1].c_str())) < 0 )
	{
		std::cerr<<"hdf5: unable to open the requested data set "<<file_params[1]<<std::endl;
		return false;
	}

	if (( _datatype = H5Dget_type(_dataset_id)) < 0)
	{
		std::cerr<<"hdf5: unable to data set type"<<std::endl;
		return false;
	}

	if ((_spaceid    = H5Dget_space(_dataset_id)) < 0)
	{
		std::cerr<<"hdf5: unable to open the requested data space"<<std::endl;
		return false;
	}

	if ((_ndim       = H5Sget_simple_extent_dims (_spaceid, _dims, NULL)) < 0)
	{
		std::cerr<<"hdf5: handling file"<<std::endl;
		return false;
	}

	if (file_params.size() == 2)
	{
		_xGrid = "";
		_yGrid = "";
		_zGrid = "";
	}
	else
	{
		_xGrid = file_params[2];
		_yGrid = file_params[3];
		_zGrid = file_params[4];
	}

	_isOpen = true;
	return true;
}

hdf5File::~hdf5File()
{
}

void hdf5File::close()
{
	if (!_isOpen)
		return;

	herr_t      status;

	if ((status = H5Dclose(_dataset_id)) < 0)
	{
		std::cerr<<"hdf5: unable to close the data set"<<std::endl;
	}


	if ((status = H5Fclose(_file_id)) < 0);
	{
		std::cerr<<"hdf5: unable to close the file"<<std::endl;
		/*
		 * XXX cduelo: No entiendo porque falla al cerrar el fichero....
		 *
		 */
	}
}

vmml::vector<3, int> hdf5File::getRealDimension()
{
	return vmml::vector<3, int>(_dims[0],_dims[1],_dims[2]);
}

bool hdf5File::readGrid(int ngrid, float ** grid)
{
	std::string name;
	hsize_t dim = 0;
	if (ngrid == 0)
	{
		dim = _dims[0];
		name = _xGrid;
	}
	else if (ngrid == 1)
	{
		dim = _dims[1];
		name = _yGrid;
	}
	else
	{
		dim = _dims[2];
		name = _zGrid;
	}

	if (name == "")
	{
		(*grid) = new float[dim];
		for(hsize_t i=0; i<dim; i++)
			(*grid)[i] = (float)i;
	}
	else
	{
		double * g = new double[dim];

		hid_t           dataset_id;
		hid_t           spaceid;
		hid_t			datatype;
		herr_t	status;

		if ((dataset_id = H5Dopen1(_file_id, name.c_str())) < 0 )
		{
			delete [] g;
			std::cerr<<"hdf5: unable to open the requested data set "<<name<<std::endl;
			return false;
		}

		if (( datatype = H5Dget_type(dataset_id)) < 0)
		{
			std::cerr<<"hdf5: unable to data set type"<<std::endl;
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				std::cerr<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		if ((spaceid    = H5Dget_space(dataset_id)) < 0)
		{
			delete[] g;
			std::cerr<<"hdf5: unable to open the requested data space"<<std::endl;
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				std::cerr<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		int ndim = 0;
		hsize_t dims[3];
		if ((ndim       = H5Sget_simple_extent_dims (spaceid, dims, NULL)) < 0)
		{
			delete[] g;
			std::cerr<<"hdf5: handling file"<<std::endl;
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				std::cerr<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}
		if (ndim != 1 || dims[0] != dim)
		{
			std::cerr<<"hdf5: x grid dimension no equal to data dimension"<<std::endl;
			delete[] g;
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				std::cerr<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		if ((status = H5Dread(dataset_id, H5T_IEEE_F64LE/*datatype*/, H5S_ALL, spaceid, H5P_DEFAULT, g)) < 0)
		{
			delete[] g;
			std::cerr<<"hdf5: reading x grid"<<std::endl;
			if ((status = H5Dclose(dataset_id)) < 0)
			{
				std::cerr<<"hdf5: unable to close the data set"<<std::endl;
				return false;
			}
			return false;
		}

		if ((status = H5Dclose(dataset_id)) < 0)
		{
			delete[] g;
			std::cerr<<"hdf5: unable to close the data set"<<std::endl;
			return false;
		}
		(*grid) = new float[dim];
		for(hsize_t i=0; i<dim; i++)
			(*grid)[i] = (float)g[i];
		delete[] g;
	}

	return true;
}

bool hdf5File::getxGrid(float ** xGrid)
{
	return readGrid(0, xGrid);
}

bool hdf5File::getyGrid(float ** yGrid)
{
	return readGrid(1, yGrid);
}

bool hdf5File::getzGrid(float ** zGrid)
{
	return readGrid(2, zGrid);
}


bool hdf5File::readPlane(float * cube, vmml::vector<3, int> s, vmml::vector<3, int> e)
{
	#ifdef DISK_TIMING
	lunchbox::Clock     timingC; 
	timingC.reset();
	#endif

	hsize_t dim[3] = {1, (hsize_t)abs(e.y()-s.y()), (hsize_t)abs(e.z()-s.z())};

	#ifdef DISK_TIMING
	lunchbox::Clock     timing; 
	timing.reset();
	#endif
	// Set zeros's
	bzero(cube, dim[0]*dim[1]*dim[2]*sizeof(float));
	#ifdef DISK_TIMING
	double time = timing.getTimed(); 
	std::cerr<<"Inicializate cube time: "<<time/1000.0<<" seconds."<<std::endl;
	#endif

	// The data required is completly outside of the dataset
	if (s.x() >= (int)this->_dims[0] || s.y() >= (int)this->_dims[1] || s.z() >= (int)this->_dims[2] || e.x() < 0 || e.y() < 0 || e.z() < 0)
	{
		std::cerr<<"Warning: reading cube outsite the volume "<<std::endl;
		std::cerr<<"Dimension volume "<<this->_dims[0]<<" "<<this->_dims[1]<<" "<<this->_dims[2]<<std::endl;
		std::cerr<<"start "<<s.x()<<" "<<s.y()<<" "<<s.z()<<std::endl;
		std::cerr<<"end "<<e.x()<<" "<<e.y()<<" "<<e.z()<<std::endl;
		std::cerr<<"Dimension plane "<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<std::endl;

		return true;
	}

	herr_t	status;
	hid_t	memspace; 
	hsize_t offset_out[3] 	= {(hsize_t)s.x() < 0 ? (hsize_t)abs(s.x()) : 0, s.y() < 0 ? (hsize_t)abs(s.y()) : 0, (hsize_t)s.z() < 0 ? (hsize_t)abs(s.z()) : 0};
	hsize_t offset[3] 	= {s.x() < 0 ? 0 : (hsize_t)s.x(), s.y() < 0 ? 0 : (hsize_t)s.y(), s.z() < 0 ? 0 : (hsize_t)s.z()};
	hsize_t dimR[3]		= {1,
				   e.y() > (int)this->_dims[1] ? this->_dims[1] - offset[1] : e.y() - offset[1],
				   e.z() > (int)this->_dims[2] ? this->_dims[2] - offset[2] : e.z() - offset[2]};

	#if 0
	std::cerr<<"Dimension cube "<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<std::endl;
	std::cerr<<"Dimension hyperSlab "<<dimR[0]<<" "<<dimR[1]<<" "<<dimR[2]<<std::endl;
	std::cerr<<"Offset in "<<offset[0]<<" "<<offset[1]<<" "<<offset[2]<<std::endl;
	std::cerr<<"Offset out "<<offset_out[0]<<" "<<offset_out[1]<<" "<<offset_out[2]<<std::endl;
	#endif
    
	#ifdef DISK_TIMING
	timing.reset();
	#endif
	// Set zeros's
	/* 
	* Define hyperslab in the dataset. 
	*/
	if ((status = H5Sselect_hyperslab(_spaceid, H5S_SELECT_SET, offset, NULL, dimR, NULL)) < 0)
	{
		std::cerr<<"hdf5: defining hyperslab in the dataset"<<std::endl;
		return false;
	}
	#ifdef DISK_TIMING
	time = timing.getTimed(); 
	std::cerr<<"Define hyperslab  time: "<<time/1000.0<<" seconds."<<std::endl;
	#endif

	#ifdef DISK_TIMING
	timing.reset();
	#endif
	/*
	* Define the memory dataspace.
	*/
	if ((memspace = H5Screate_simple(3, dim, NULL)) < 0)
	//if ((memspace = H5Screate_simple(3, dimR, NULL)) < 0)
	{
		std::cerr<<"hdf5: defining the memory space"<<std::endl;
		return false;
	}
	#ifdef DISK_TIMING
	time = timing.getTimed(); 
	std::cerr<<"Define memorydataset time: "<<time/1000.0<<" seconds."<<std::endl;
	#endif


	#ifdef DISK_TIMING
	timing.reset();
	#endif
	/* 
	* Define memory hyperslab. 
	*/
	if ((status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL, dimR, NULL)) < 0)
	{
		std::cerr<<"hdf5: defining the memory hyperslab"<<std::endl;
		return false;
	}
	#ifdef DISK_TIMING
	time = timing.getTimed(); 
	std::cerr<<"Define memory hyperslab time: "<<time/1000.0<<" seconds."<<std::endl;
	#endif

	#ifdef DISK_TIMING
	timing.reset();
	#endif
	/*
	* Read data from hyperslab in the file into the hyperslab in 
	* memory and display.
	*/
	if ((status = H5Dread(_dataset_id, H5T_IEEE_F32LE/*_datatype*/, memspace, _spaceid, H5P_DEFAULT, cube)) < 0)
	{
		std::cerr<<"hdf5: reading data from hyperslab un the file"<<std::endl;
		return false;
	}
	#ifdef DISK_TIMING
	time = timing.getTimed(); 
	std::cerr<<"Reading time: "<<time/1000.0<<" seconds."<<std::endl;
	#endif


	#ifdef DISK_TIMING
	timing.reset();
	#endif
	if ((status = H5Sclose(memspace)) < 0)
	{
		std::cerr<<"hdf5: closing dataspace"<<std::endl;
		return false;
	}
	#ifdef DISK_TIMING
	time = timing.getTimed(); 
	std::cerr<<"Close dataspace time: "<<time/1000.0<<" seconds."<<std::endl;
	double timeC = timingC.getTimed(); 
	std::cerr<<"Read in MB: "<<(dimR[0]*dimR[1]*dimR[2]*sizeof(float)/1024.f/1024.f)<<" in "<<(timeC/1000.0f)<<" seconds."<<std::endl;
	std::cerr<<"Bandwidth: "<<(dimR[0]*dimR[1]*dimR[2]*sizeof(float)/1024.f/1024.f)/(timeC/1000.0f)<<" MB/seconds."<<std::endl;
	#endif
	
	return true;
}

void hdf5File::readCube(index_node_t index, float * cube, int levelCube, int nLevels, vmml::vector<3, int>    cubeDim, vmml::vector<3, int> offsetCube)
{
	#ifdef DISK_TIMING
	lunchbox::Clock     timingC; 
	timingC.reset();
	#endif

	vmml::vector<3, int> coord 	= getMinBoxIndex2(index, levelCube, nLevels);
	// ADD offset
	coord += offsetCube;
	vmml::vector<3, int> s 		= coord - vmml::vector<3, int>(CUBE_INC, CUBE_INC, CUBE_INC);
	vmml::vector<3, int> e 		= s + cubeDim;

	hsize_t dim[3] = {(hsize_t)abs(e.x()-s.x()),(hsize_t)abs(e.y()-s.y()),(hsize_t)abs(e.z()-s.z())};

	#ifdef DISK_TIMING
	lunchbox::Clock     timing; 
	timing.reset();
	#endif
	// Set zeros's
	bzero(cube, dim[0]*dim[1]*dim[2]*sizeof(float));
	#ifdef DISK_TIMING
	double time = timing.getTimed(); 
	std::cerr<<"Inicializate cube time: "<<time/1000.0<<" seconds."<<std::endl;
	#endif

	// The data required is completly outside of the dataset
	if (s.x() >= (int)this->_dims[0] || s.y() >= (int)this->_dims[1] || s.z() >= (int)this->_dims[2] || e.x() < 0 || e.y() < 0 || e.z() < 0)
	{
		std::cerr<<"Warning: reading cube outsite the volume "<<std::endl;
		std::cerr<<"Dimension volume "<<this->_dims[0]<<" "<<this->_dims[1]<<" "<<this->_dims[2]<<std::endl;
		std::cerr<<"Cube dimension "<<cubeDim<<" in level "<<levelCube<<std::endl;
		std::cerr<<"Dimension volume "<<this->_dims[0]<<" "<<this->_dims[1]<<" "<<this->_dims[2]<<std::endl;
		std::cerr<<"Index "<<index<<" Coordinate "<<coord<<std::endl;
		std::cerr<<"start "<<s.x()<<" "<<s.y()<<" "<<s.z()<<std::endl;
		std::cerr<<"end "<<e.x()<<" "<<e.y()<<" "<<e.z()<<std::endl;
		std::cerr<<"Dimension cube "<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<std::endl;

		return;
	}

	herr_t	status;
	hid_t	memspace; 
	hsize_t offset_out[3] 	= {s.x() < 0 ? (hsize_t)abs(s.x()) : 0, s.y() < 0 ? (hsize_t)abs(s.y()) : 0, s.z() < 0 ? (hsize_t)abs(s.z()) : 0};
	hsize_t offset[3] 	= {s.x() < 0 ? 0 : (hsize_t)s.x(), s.y() < 0 ? 0 : (hsize_t)s.y(), s.z() < 0 ? 0 : (hsize_t)s.z()};
	hsize_t dimR[3]		= {e.x() > (int)this->_dims[0] ? this->_dims[0] - offset[0] : e.x() - offset[0],
				   e.y() > (int)this->_dims[1] ? this->_dims[1] - offset[1] : e.y() - offset[1],
				   e.z() > (int)this->_dims[2] ? this->_dims[2] - offset[2] : e.z() - offset[2]};

	#ifdef DISK_TIMING
	timing.reset();
	#endif
	// Set zeros's
	/* 
	* Define hyperslab in the dataset. 
	*/
	if ((status = H5Sselect_hyperslab(_spaceid, H5S_SELECT_SET, offset, NULL, dimR, NULL)) < 0)
	{
		std::cerr<<"hdf5: defining hyperslab in the dataset"<<std::endl;
	}
	#ifdef DISK_TIMING
	time = timing.getTimed(); 
	std::cerr<<"Define hyperslab  time: "<<time/1000.0<<" seconds."<<std::endl;
	#endif

	#ifdef DISK_TIMING
	timing.reset();
	#endif
	/*
	* Define the memory dataspace.
	*/
	if ((memspace = H5Screate_simple(3, dim, NULL)) < 0)
	//if ((memspace = H5Screate_simple(3, dimR, NULL)) < 0)
	{
		std::cerr<<"hdf5: defining the memory space"<<std::endl;
	}
	#ifdef DISK_TIMING
	time = timing.getTimed(); 
	std::cerr<<"Define memorydataset time: "<<time/1000.0<<" seconds."<<std::endl;
	#endif


	#ifdef DISK_TIMING
	timing.reset();
	#endif
	/* 
	* Define memory hyperslab. 
	*/
	if ((status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL, dimR, NULL)) < 0)
	{
		std::cerr<<"hdf5: defining the memory hyperslab"<<std::endl;
	}
	#ifdef DISK_TIMING
	time = timing.getTimed(); 
	std::cerr<<"Define memory hyperslab time: "<<time/1000.0<<" seconds."<<std::endl;
	#endif

	#ifdef DISK_TIMING
	timing.reset();
	#endif
	/*
	* Read data from hyperslab in the file into the hyperslab in 
	* memory and display.
	*/
	if ((status = H5Dread(_dataset_id, H5T_IEEE_F32LE/*_datatype*/, memspace, _spaceid, H5P_DEFAULT, cube)) < 0)
	{
		std::cerr<<"hdf5: reading data from hyperslab un the file"<<std::endl;
	}
	#ifdef DISK_TIMING
	time = timing.getTimed(); 
	std::cerr<<"Reading time: "<<time/1000.0<<" seconds."<<std::endl;
	#endif


	#ifdef DISK_TIMING
	timing.reset();
	#endif
	if ((status = H5Sclose(memspace)) < 0)
	{
		std::cerr<<"hdf5: closing dataspace"<<std::endl;
	}
	#ifdef DISK_TIMING
	time = timing.getTimed(); 
	std::cerr<<"Close dataspace time: "<<time/1000.0<<" seconds."<<std::endl;
	double timeC = timingC.getTimed(); 
	std::cerr<<"Read in MB: "<<(dimR[0]*dimR[1]*dimR[2]*sizeof(float)/1024.f/1024.f)<<" in "<<(timeC/1000.0f)<<" seconds."<<std::endl;
	std::cerr<<"Bandwidth: "<<(dimR[0]*dimR[1]*dimR[2]*sizeof(float)/1024.f/1024.f)/(timeC/1000.0f)<<" MB/seconds."<<std::endl;
	#endif
}
}
