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

#define CUBE		(unsigned char)0b00010000
#define PAINTED		(unsigned char)0b00001000
#define CACHED		(unsigned char)0b00000100
#define NOCACHED	(unsigned char)0b00000010
#define NOCUBE		(unsigned char)0b00000001
#define NONE		(unsigned char)0b00000000

#define CUDA_CUBE		(unsigned char)0x00F
#define CUDA_PAINTED	(unsigned char)0x008
#define CUDA_CACHED		(unsigned char)0x004
#define CUDA_NOCACHED	(unsigned char)0x002
#define CUDA_NOCUBE		(unsigned char)0x001
#define CUDA_NONE		(unsigned char)0x000

#define NUM_COLORS 256
#define COLOR_INC 0.00390625f

#define CUBE_INC 2

}
#endif /* EQ_MIVT_TYPEDEF_H */
