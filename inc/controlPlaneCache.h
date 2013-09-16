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
				return t1.timestamp < t2.timestamp;
			else
				return t1.refs != 0;
		}
};

class  ControlPlaneCache : public lunchbox::Thread
{
	private:
		std::priority_queue<cache_plane_t, std::vector<cache_plane_t>, ComparePlane>	_lruPlanes;

		boost::unordered_map<int, cache_plane_t *>	_currentPlanes;
		std::queue<int>								_pendingPlanes;
		
		int _maxNumPlanes;
		int	_maxPlane;
		float *	_memoryPlane;
		int		_sizePlane;

		lunchbox::Lock		_lockEnd;
		bool				_end;
		lunchbox::Lock		_lock;
		lunchbox::Condition	_emptyPendingPlanes;

		bool readPlane(float * data, int plane);

	public:

		virtual ~ControlPlaneCache() {}

		virtual void run();

		virtual bool init();

		void stopProcessing();
		
		virtual void exit() {}

		void addPlane(int plane);
		void addPlanes(std::vector<int> planes);

		float * getAndBlockPlane(int plane);
		void	unlockPlane(int plane);

};

}

#endif /*EQ_MIVT_CONTROL_PLANE_CACHE_H*/
