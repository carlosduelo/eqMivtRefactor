/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_HDF5_FILE_H
#define EQ_MIVT_HDF5_FILE_H

#include <hdf5.h>

//STL
#include <string>

#include <vmmlib/vector.hpp>

namespace eqMivt
{
class hdf5File
{
	private:
		// HDF5 stuff
		hid_t           _file_id;
		hid_t           _dataset_id;
		hid_t           _spaceid;
		int             _ndim;
		hsize_t         _dims[3];
		hid_t			_datatype;

		std::string		_xGrid;
		std::string		_yGrid;
		std::string		_zGrid;

	public:

		bool init(std::vector<std::string> file_params);

		~hdf5File();

		bool getxGrid(double ** xGrid);
		bool getyGrid(double ** yGrid);
		bool getzGrid(double ** zGrid);

		void readPlane(float * cube, vmml::vector<3, int> s, vmml::vector<3, int> e);

		vmml::vector<3, int> getRealDimension();
};
}

#endif /* EQ_MIVT_HDF5_FILE */
