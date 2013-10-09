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

		indexVisibleCubeGPU_t getIndexVisibleCubesGPU();

		visibleCubeGPU_t getVisibleCubesGPU();

		int getSizeGPU();

		int getSize();

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

			int						_sizeGPU;
			indexVisibleCubeGPU_t	_indexGPU;
			visibleCubeGPU_t		_visibleCubesGPU;
};

}
#endif /*EQ_MIVT_VISIBLE_CUBES_H*/
