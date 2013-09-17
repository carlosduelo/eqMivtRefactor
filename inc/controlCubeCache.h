/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_CONTROL_CUBE_CACHE_H
#define EQ_MIVT_CONTROL_CUBE_CACHE_H

#include <vector>
#include <boost/unordered_map.hpp>
#include<ctime>

#include <lunchbox/thread.h>
#include <lunchbox/lock.h>
#include <lunchbox/condition.h>

namespace eqMivt
{

class ControlCubeCache
{
struct cache_cube_t
{
	index_node_t id;
	float * data;
	int		refs;
	std::time_t timestamp;
};

};

}

#endif /*EQ_MIVT_CONTROL_CUBE_CACHE_H*/

