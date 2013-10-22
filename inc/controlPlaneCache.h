/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CONTROL_PLANE_CACHE_H
#define EQ_MIVT_CONTROL_PLANE_CACHE_H

#include <controlCache.h>
#include <hdf5File.h>
#include <linkedList.h>

//STL
#include <vector>
#include <boost/unordered_map.hpp>
#include<ctime>

namespace eqMivt
{

class  ControlPlaneCache : public ControlCache 
{
	private:
		LinkedList									_lruPlanes;

		boost::unordered_map<int, NodeLinkedList *>	_currentPlanes;
		std::vector<int>							_pendingPlanes;
		int											_lastPlane;
		
		double	_memoryAviable;
		int		_freeSlots;
		int		_maxNumPlanes;
		float *	_memoryPlane;
		int		_sizePlane;
		float	_memoryOccupancy;

		lunchbox::Condition	_emptyPendingPlanes;
		lunchbox::Condition	_fullSlots;

		vmml::vector<3, int> _min;
		vmml::vector<3, int> _max;
		vmml::vector<3, int> _minFuture;
		vmml::vector<3, int> _maxFuture;

		std::vector<std::string> _file_parameters;
		hdf5File	_file;

		bool readPlane(float * data, int plane);

		virtual void _threadWork();

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

		float * getAndBlockPlane(int plane);

		void	unlockPlane(int plane);

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
