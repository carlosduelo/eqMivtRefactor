/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_RENDER_H
#define EQ_MIVT_RENDER_H

#include <octree.h>
#include <cache.h>
#include <queue>
#include <lunchbox/condition.h>
#include <lunchbox/thread.h>

#include <cuda_help.h>

namespace eqMivt
{

class Render
{
	protected:
		float	*		_pixelBuffer;

		int _pvpW;
		int _pvpH;

		/* VISIBLE CUBES */
		int						_size;
		visibleCube_t *			_visibleCubes;
		visibleCubeGPU_t		_visibleCubesGPU;

		cudaStream_t			_stream;
		Cache			_cache;

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

		bool cacheIsInit() { return _cache.isInit();}

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
