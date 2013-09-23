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
#include <set>

#include <cuda_runtime.h>

namespace eqMivt
{

typedef struct
{
	index_node_t	id;
	float *			data;
	statusCube		state;
	index_node_t	cubeID;
	int				pixel;
} visibleCube_t;

typedef visibleCube_t * visibleCubeGPU;

class VisibleCubes
{
	public:

		VisibleCubes();

		void  init();

		void reSize(int numPixels);

		void updateCPU();

		void updateGPU(statusCube type, bool sync, cudaStream_t stream);

		visibleCubeGPU getVisibleCubesGPU();

		int getSizeGPU();

		std::vector<int> getListCubes(statusCube type);

		void updateVisibleCubes(std::vector<visibleCube_t> list);

		visibleCube_t getCube(int i);

		int getNumElements(statusCube type);

		private:
			int					_size;
			visibleCube_t *		_visibleCubes;
			visibleCube_t *		_visibleCubesAUX;

			std::set<int>	_cube;
			std::set<int>	_painted;
			std::set<int>	_cached;
			std::set<int>	_nocube;

			int					_sizeGPU;
			visibleCubeGPU		_visibleCubesGPU;
		
			void updateCube(int iter, int idCube, statusCube state, float * data);
};

}
#endif /*EQ_MIVT_VISIBLE_CUBES_H*/
