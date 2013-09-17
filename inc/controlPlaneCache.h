/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CONTROL_PLANE_CACHE_H
#define EQ_MIVT_CONTROL_PLANE_CACHE_H

//STL
#include <queue>
#include <boost/unordered_map.hpp>
#include<ctime>

#include <lunchbox/thread.h>
#include <lunchbox/lock.h>
#include <lunchbox/condition.h>

#include <hdf5File.h>

namespace eqMivt
{

typedef struct
{
	int		id;
	float * data;
	int		refs;
	std::time_t timestamp;
} cache_plane_t;

class ComparePlane
{
	public:
	    bool operator()(cache_plane_t& t1, cache_plane_t& t2)
		{
			if (t1.refs > 0 && t2.refs > 0)
				return t1.timestamp > t2.timestamp;
			else if (t1.refs == -1)
				return true;
			else if (t2.refs == -1)
				return false;
			else
				return t1.refs == 0;
		}
};

class  ControlPlaneCache : public lunchbox::Thread
{
	private:
		std::priority_queue<cache_plane_t, std::vector<cache_plane_t>, ComparePlane>	_lruPlanes;

		boost::unordered_map<int, cache_plane_t>	_currentPlanes;
		std::vector<int>							_pendingPlanes;
		
		int		_freeSlots;
		int		_maxNumPlanes;
		int		_maxPlane;
		float *	_memoryPlane;
		int		_sizePlane;

		lunchbox::Lock		_lockEnd;
		bool				_end;

		lunchbox::Lock		_currentPlanesLock;
		lunchbox::Condition	_emptyPendingPlanes;
		lunchbox::Condition	_fullSlots;

		hdf5File	_file;

		bool readPlane(float * data, int plane);

	public:

		virtual ~ControlPlaneCache();

		virtual void run();

		bool initParamenter(std::vector<std::string> file_parameters, int maxHeight);

		void stopProcessing();

		float * getAndBlockPlane(int plane);
		void	unlockPlane(int plane);

};

}

#endif /*EQ_MIVT_CONTROL_PLANE_CACHE_H*/
