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
typedef int *			indexVisibleCubeGPU;

class VisibleCubes
{
	public:

		VisibleCubes();

		~VisibleCubes();

		void  init();

		void reSize(int numPixels);

		void destroy();

		void updateIndexCPU();

		void updateCPU();

		void updateGPU(statusCube type, cudaStream_t stream);

		indexVisibleCubeGPU getIndexVisibleCubesGPU();

		visibleCubeGPU getVisibleCubesGPU();

		int getSizeGPU();

		std::vector<int> getListCubes(statusCube type);

		visibleCube_t * getCube(int i);

		int getNumElements(statusCube type);

		private:
			int					_size;
			visibleCube_t *		_visibleCubes;

			int * _cubeC;
			int	* _paintedC;
			int	* _cachedC;
			int	* _nocubeC;
			int * _cubeG;
			int	* _paintedG;
			int	* _cachedG;
			int	* _nocubeG;

			int					_sizeGPU;
			indexVisibleCubeGPU	_indexGPU;
			visibleCubeGPU		_visibleCubesGPU;
};

}
#endif /*EQ_MIVT_VISIBLE_CUBES_H*/
