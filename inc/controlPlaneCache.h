/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CONTROL_PLANE_CACHE_H
#define EQ_MIVT_CONTROL_PLANE_CACHE_H

//STL
#include <vector>
#include <boost/unordered_map.hpp>
#include<ctime>

#include <lunchbox/thread.h>
#include <lunchbox/lock.h>
#include <lunchbox/condition.h>

#include <hdf5File.h>

namespace eqMivt
{

struct cache_plane_t
{
	int		id;
	float * data;
	int		refs;
	std::time_t timestamp;
};

class  ControlPlaneCache : public lunchbox::Thread
{
	private:
		cache_plane_t	*							_cachePlanes;
		std::vector<cache_plane_t *>				_lruPlanes;

		boost::unordered_map<int, cache_plane_t *>	_currentPlanes;
		std::vector<int>							_pendingPlanes;
		
		int		_freeSlots;
		int		_maxNumPlanes;
		float *	_memoryPlane;
		int		_sizePlane;

		lunchbox::Lock		_lockEnd;
		bool				_end;

		lunchbox::Lock		_currentPlanesLock;
		lunchbox::Condition	_emptyPendingPlanes;
		lunchbox::Condition	_fullSlots;

		vmml::vector<3, int> _min;
		vmml::vector<3, int> _max;

		hdf5File	_file;

		bool readPlane(float * data, int plane);

	public:

		virtual ~ControlPlaneCache();

		virtual void run();

		/* Read planes from [min,max) */

		bool initParameter(std::vector<std::string> file_parameters, vmml::vector<3, int> min, vmml::vector<3, int> max);

		void stopProcessing();

		float * getAndBlockPlane(int plane);

		void	unlockPlane(int plane);

		/* (x,y) = (y_dim, z_dim) */
		vmml::vector<2,int>	getPlaneDim();
		vmml::vector<3,int> getMinCoord() { return _min; }
		vmml::vector<3,int> getMaxCoord() { return _max; }

		int	getMaxPlane() { return _max.x(); }
		int	getMinPlane() { return _min.x(); }

};

}

#endif /*EQ_MIVT_CONTROL_PLANE_CACHE_H*/
