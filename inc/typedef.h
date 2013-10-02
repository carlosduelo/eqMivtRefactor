/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#ifndef EQ_MIVT_TYPEDEF_H
#define EQ_MIVT_TYPEDEF_H

namespace eqMivt
{

/* indentifier type for octree's node */
typedef unsigned long long index_node_t;

typedef unsigned char statusCube;

typedef int	device_t;

#define CUBE		(eqMivt::statusCube)0b00001000
#define PAINTED		(eqMivt::statusCube)0b00000100
#define CACHED		(eqMivt::statusCube)0b00000010
#define NOCUBE		(eqMivt::statusCube)0b00000001
#define NONE		(eqMivt::statusCube)0b00000000

#define CUDA_CUBE		(eqMivt::statusCube)0x008
#define CUDA_PAINTED	(eqMivt::statusCube)0x004
#define CUDA_CACHED		(eqMivt::statusCube)0x002
#define CUDA_NOCUBE		(eqMivt::statusCube)0x001
#define CUDA_NONE		(eqMivt::statusCube)0x000

#define NUM_COLORS 256
#define COLOR_INC 0.00390625f

#define CUBE_INC 2

#define RUNNING 4
#define STOPPED 3
#define PAUSED  2
#define STARTED 1

}
#endif /* EQ_MIVT_TYPEDEF_H */
