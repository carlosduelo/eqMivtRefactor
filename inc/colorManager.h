/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

*/

#ifndef EQ_MIVT_COLOR_MANAGER_H
#define EQ_MIVT_COLOR_MANAGER_H

#include <typedef.h>
#include <boost/unordered_map.hpp>

namespace eqMivt
{

class ColorManager
{
	private:
		float *	_colorsC;
		boost::unordered_map<device_t, color_t> _colors;

	public:
		bool init(std::string file_name);

		void destroy();

		color_t getColors(device_t device);
};


}
#endif /* EQ_MIVT_COLOR_MANAGER_H */
