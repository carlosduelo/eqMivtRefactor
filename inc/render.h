/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_RENDER_H
#define EQ_MIVT_RENDER_H

#include <visibleCubes.h>
#include <octree.h>
#include <cache.h>

namespace eqMivt
{

class Render
{
	protected:
		float	*		_pixelBuffer;

		int _pvpW;
		int _pvpH;

	private:
		VisibleCubes	_vC;
		Cache			_cache;
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

	public:
		virtual ~Render(){};

		virtual bool init(device_t device);

		virtual void destroy();

		device_t getDevice(){ return _device; }

		virtual bool setViewPort(int pvpW, int pvpH);

		bool cacheIsInit() { return _cache.isInit(); }

		bool octreeIsInit() { return _octree != 0; }

		bool colorsIsInit() { return _colors.r != 0; }

		bool setCache(ControlCubeCache * ccc);

		void setRayCastingLevel(int rLevel){ _cache.setRayCastingLevel(rLevel); }

		void setOctree(Octree * octree){ _octree = octree; }

		void setColors(color_t colors){ _colors = colors; }

		void setDrawCubes(bool drawCubes){ _drawCube = drawCubes; }

		virtual bool frameDraw(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
								vmml::vector<4, float> up, vmml::vector<4, float> right,
								float w, float h);
};

}


#endif /*EQ_MIVT_RENDER_H*/
