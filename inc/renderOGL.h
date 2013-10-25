/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_RENDER_OGL_H
#define EQ_MIVT_RENDER_OGL_H

#include <render.h>
#include <cuda_gl_interop.h> 
//#include "cuda_runtime.h"

namespace eqMivt
{

class RenderOGL : public Render
{
	private:
		std::string _name;
		int			_frame;
  	    struct cudaGraphicsResource * _cuda_pbo_resource;
	public:
		virtual bool init(device_t device, std::string name);

		virtual bool setViewPort(int pvpW, int pvpH, GLuint pbo);

		virtual void destroy();

		virtual bool frameDraw(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
								vmml::vector<4, float> up, vmml::vector<4, float> right,
								float w, float h);
};

}
#endif /* EQ_MIVT_RENDER_OGL_H */

