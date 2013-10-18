/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_RENDER_PNG_H
#define EQ_MIVT_RENDER_PNG_H

#include <render.h>

namespace eqMivt
{

class RenderPNG : public Render
{
	private:
		std::string _name;
		int			_frame;
		float *		_bufferC;
	public:
		virtual bool init(device_t device, std::string name);

		virtual bool setViewPort(int pvpW, int pvpH);

		virtual void destroy();

		virtual bool frameDraw(	vmml::vector<4, float> origin, vmml::vector<4, float> LB,
								vmml::vector<4, float> up, vmml::vector<4, float> right,
								float w, float h);
};

}
#endif /* EQ_MIVT_RENDER_PNG_H */
