/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_VISIBLE_CUBES_H
#define EQ_MIVT_VISIBLE_CUBES_H

#include <typedef.h>

//STL
#include <vector>

#include <cuda_runtime.h>

namespace eqMivt
{

typedef struct
{
	index_node_t	id;
	float *			data;
	unsigned char   state;
	index_node_t	cubeID;
	int				pixel;
} visibleCube_t;

typedef struct
{
	index_node_t	cubeID;
	int				pixel;
	unsigned char	state;

} updateCube_t;


typedef visibleCube_t * visibleCubeGPU;

class VisibleCubes
{
	public:

		VisibleCubes();

		void  init();

		void reSize(int numPixels);

		void updateCPU();

		void updateGPU(unsigned char type, bool sync, cudaStream_t stream);

		visibleCubeGPU getVisibleCubesGPU();

		int getSizeGPU();

		std::vector<int> getListCubes(unsigned char type);

		void updateVisibleCubes(std::vector<updateCube_t> list);

		visibleCube_t getCube(int i);

		private:
			int					_size;
			visibleCube_t *		_visibleCubes;
			visibleCube_t *		_visibleCubesAUX;

			std::vector<int>	_cube;
			std::vector<int>	_painted;
			std::vector<int>	_cached;
			std::vector<int>	_nocached;
			std::vector<int>	_nocube;

			int					_sizeGPU;
			visibleCubeGPU		_visibleCubesGPU;
		
			void updateCube(int iter, int idCube, int state);
};

}
#endif /*EQ_MIVT_VISIBLE_CUBES_H*/
