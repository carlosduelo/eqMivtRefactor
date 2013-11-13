/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_RENDER_H
#define EQ_MIVT_RENDER_H

#include <renderWorkers.h>

#define MAX_WORKERS 4

namespace eqMivt
{

class Render
{
	protected:
		float	*		_pixelBuffer;

		int _pvpW;
		int _pvpH;
		sharedParameters_t 	_parameters;

		/* VISIBLE CUBES */
		int						_size;
		visibleCube_t *			_visibleCubes;
		visibleCubeGPU_t		_visibleCubesGPU;

		/* WORKERS */
		queue_t		_octreeQueue[MAX_WORKERS];
		queue_t		_cacheQueue[MAX_WORKERS];
		queue_t		_rayCasterQueue[MAX_WORKERS];
		queue_t		_masterQueue;
		queuePOP_t	_popQueue[MAX_WORKERS];
		workerOctree	_octreeWorkers[MAX_WORKERS];
		workerCache		_cacheWorkers[MAX_WORKERS];
		workerRayCaster _rayCasterWorkers[MAX_WORKERS];
		workerPoper		_poperWorkers[MAX_WORKERS];

	private:
		ControlCubeCache * _ccc;
		Octree *		_octree;
		color_t			_colors;

		bool			_drawCube;

		device_t		_device;

		bool _draw(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
						vmml::vector<4, float> up, vmml::vector<4, float> right,
						float w, float h);
		bool _drawCubes(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
						vmml::vector<4, float> up, vmml::vector<4, float> right,
						float w, float h);

		void _destroyVisibleCubes();

	public:
		virtual ~Render(){};

		virtual bool init(device_t device);

		virtual void destroy();

		device_t getDevice(){ return _device; }

		virtual bool setViewPort(int pvpW, int pvpH);

		bool cacheIsInit() { return _ccc != 0;}

		bool octreeIsInit() { return _octree != 0; }

		bool colorsIsInit() { return _colors.r != 0; }

		bool setCache(ControlCubeCache * ccc);

		void setOctree(Octree * octree);

		void setColors(color_t colors);

		void setDrawCubes(bool drawCubes){ _drawCube = drawCubes; }

		virtual bool frameDraw(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
								vmml::vector<4, float> up, vmml::vector<4, float> right,
								float w, float h);
};

}


#endif /*EQ_MIVT_RENDER_H*/
