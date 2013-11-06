/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_TYPEDEF_H
#define EQ_MIVT_TYPEDEF_H

#include <vmmlib/vector.hpp>

namespace eqMivt
{

/* indentifier type for octree's node */
typedef unsigned long long index_node_t;

typedef unsigned char statusCube;

typedef int	device_t;

typedef struct
{
	index_node_t	id;
	float *			data;
	statusCube		state;
} visibleCube_t;

typedef visibleCube_t * visibleCubeGPU_t;
typedef int *			indexVisibleCubeGPU_t;

struct color_t
{
	float * r;
	float * g;
	float * b;
};

struct octreePosition_t
{
	vmml::vector<3, int>	start;
	vmml::vector<3, int>	end;
	std::vector<float>		isos;
	std::vector<int>		index;
	std::vector<int>		maxHeight;
	int						cubeLevel;
	int						rayCastingLevel;
	int						nLevels;
	int						maxLevel;

	friend std::ostream& operator<<(std::ostream& os, const octreePosition_t& o)
	{
		os<<"Octree:"<<std::endl;
		os<<o.start<<" "<<o.end<<std::endl;
		os<<"Isosurfaces ";
		for(std::vector<float>::const_iterator it=o.isos.begin(); it!=o.isos.end(); it++)
			os<<*it<<" ";
		os<<std::endl;
		os<<"nLevels "<<o.nLevels<<std::endl;
		os<<"maxLevel "<<o.maxLevel<<std::endl;
		os<<"Cube Level "<<o.cubeLevel<<std::endl;
		os<<"Ray casting level "<<o.rayCastingLevel<<std::endl;
		return os;
	}
};

#define CUBE		(eqMivt::statusCube)0b00001000
#define PAINTED		(eqMivt::statusCube)0b00000100
#define CACHED		(eqMivt::statusCube)0b00000010
#define NOCUBE		(eqMivt::statusCube)0b00000001
#define EMPTY		(eqMivt::statusCube)0b00000000

#define CUDA_CUBE		(eqMivt::statusCube)0x008
#define CUDA_PAINTED	(eqMivt::statusCube)0x004
#define CUDA_CACHED		(eqMivt::statusCube)0x002
#define CUDA_NOCUBE		(eqMivt::statusCube)0x001
#define CUDA_EMPTY		(eqMivt::statusCube)0x000

#define NUM_COLORS 256
#define COLOR_INC 0.00390625f

#define CUBE_INC 2
#define MAX_LEVEL 10
#define MIN_LEVEL 7

#define MEMORY_OCCUPANCY_PLANE_CACHE 0.4*0.8
#define MEMORY_OCCUPANCY_CUBE_CACHE 0.6*0.8

#define TO3V(v) (vmml::vector<3,float>((v.x()),(v.y()),(v.z())))
}
#endif /* EQ_MIVT_TYPEDEF_H */
