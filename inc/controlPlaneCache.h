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

#include <linkedList.h>

namespace eqMivt
{

class  ControlPlaneCache : public lunchbox::Thread
{
	private:
		LinkedList									_lruPlanes;

		boost::unordered_map<int, NodeLinkedList *>	_currentPlanes;
		std::vector<int>							_pendingPlanes;
		index_node_t								_lastPlane;
		
		double	_memoryAviable;
		int		_freeSlots;
		int		_maxNumPlanes;
		float *	_memoryPlane;
		int		_sizePlane;

		lunchbox::Lock		_lockEnd;
		bool				_end;
		lunchbox::Condition	_lockResize;
		bool				_resize;

		lunchbox::Lock		_currentPlanesLock;
		lunchbox::Condition	_emptyPendingPlanes;
		lunchbox::Condition	_fullSlots;

		vmml::vector<3, int> _min;
		vmml::vector<3, int> _max;
		vmml::vector<3, int> _minFuture;
		vmml::vector<3, int> _maxFuture;

		hdf5File	_file;

		bool readPlane(float * data, int plane);

		void reSizeStructures();
	public:

		virtual ~ControlPlaneCache();

		virtual void run();
		virtual void exit();

		/* Read planes from [min,max) */
		bool initParameter(std::vector<std::string> file_parameters);

		bool reSize(vmml::vector<3,int> min, vmml::vector<3,int> max);

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
