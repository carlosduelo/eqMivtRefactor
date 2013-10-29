/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CONTROL_PLANE_CACHE_H
#define EQ_MIVT_CONTROL_PLANE_CACHE_H

#include <controlElementCache.h>
#include <hdf5File.h>

namespace eqMivt
{

class  ControlPlaneCache : public ControlElementCache<int> 
{
	private:
		double	_memoryAviable;
		int		_maxNumPlanes;
		float	_memoryOccupancy;

		vmml::vector<3, int> _min;
		vmml::vector<3, int> _max;
		vmml::vector<3, int> _minFuture;
		vmml::vector<3, int> _maxFuture;

		std::vector<std::string> _file_parameters;
		hdf5File	_file;

		virtual bool _readElement(NodeLinkedList<int> * element);

		virtual bool _threadInit();

		virtual void _threadStop();

		virtual void _freeCache();

		virtual void _reSizeCache();
	public:

		virtual ~ControlPlaneCache() {};

		/* Read planes from [min,max) */
		bool initParameter(std::vector<std::string> file_parameters, float memoryOccupancy);

		bool freeCacheAndPause();

		bool reSizeCacheAndContinue(vmml::vector<3,int> min, vmml::vector<3,int> max);

		// NO SAFE CALLS
		/* (x,y) = (y_dim, z_dim) */
		vmml::vector<2,int>	getPlaneDim() { return vmml::vector<2,int> (_max.y() - _min.y(), _max.z() - _min.z());}
		vmml::vector<3,int> getMinCoord() { return _min; }
		vmml::vector<3,int> getMaxCoord() { return _max; }

		int	getMaxPlane() { return _max.x(); }
		int	getMinPlane() { return _min.x(); }

};

}

#endif /*EQ_MIVT_CONTROL_PLANE_CACHE_H*/
